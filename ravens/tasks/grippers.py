# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from ravens.utils import pybullet_utils

import pybullet as p
import time

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'

FINGERS_BASE_URDF = 'ur5/gripper/robotiq_2f_85.urdf'


class Gripper:
  """Base gripper class."""

  def __init__(self, assets_root, ee):
    self.assets_root = assets_root
    self.activated = False
    self.parentLinkIndex = ee
    self.body = None

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


class Spatula(Gripper):
  """Simulate simple spatula for pushing."""

  def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
    """Creates spatula and 'attaches' it to the robot."""
    super().__init__(assets_root, ee)

    # Load spatula model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
    self.constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))
    self.body = base

    print('self.body/base spatula: ', self.body)
    print('constraint_id: ', self.constraint_id)

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    # print(points)
    # exit()
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False


class Suction(Gripper):
  """Simulate simple suction dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """
    super().__init__(assets_root, ee)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    self.body_visual = base
    print('base body suction (visual) id: ', base)

    # Load suction tip model (visual and collision) with compliance.
    # urdf = 'assets/ur5/suction/suction-head.urdf'
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
    print('self.body suction: ', self.body, ' ==================================')
    self.constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
    p.changeConstraint(self.constraint_id, maxForce=50)

    print('constraint_id: ', self.constraint_id)

    # Reference to object IDs in environment for simulating suction.
    self.obj_ids = obj_ids

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self):
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
      # print(points)
      if points:

        # Handle contact between suction with a rigid object.
        for point in points:
          obj_id, contact_link = point[2], point[4]
        if obj_id in self.obj_ids['rigid']:
          body_pose = p.getLinkState(self.body, 0)
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          obj_to_body = p.multiplyTransforms(world_to_body[0],
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=obj_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_body[0],
              parentFrameOrientation=obj_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))

        self.activated = True

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False

      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    # print(points)
    # exit()
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    return suctioned_object is not None



#-----------------------------------------------------------------------------
# Parallel-Jaw Two-Finger Gripper (TODO: fix)
#-----------------------------------------------------------------------------

#class Robotiq2F85:
class Fingers(Gripper):

    #def __init__(self, robot, tool):
    def __init__(self, assets_root, robot, ee, obj_ids):
        super().__init__(assets_root, ee)

        # ??????????????????????????????????????????????????????????????????????
        self.robot = robot
        #self.tool = tool # looks like ee?
        self.ee = ee
        # ??????????????????????????????????????????????????????????????????????

        # Load finger gripper model (visual only?)
        pos = [0.487, 0.109, 0.421]
        rot = p.getQuaternionFromEuler([np.pi, 0, 0])
        #urdf = 'assets/ur5/gripper/robotiq_2f_85.urdf'
        urdf = FINGERS_BASE_URDF
        self.body = p.loadURDF(os.path.join(self.assets_root, urdf), pos, rot) # base?
        self.n_joints = p.getNumJoints(self.body)
        self.activated = False

        # Connect gripper base to robot tool
        p.createConstraint(
            parentBodyUniqueId = robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=0, # may be -1?
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -0.05])

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self.body)):
            p.changeDynamics(self.body, i,
                             lateralFriction=1.5,
                             spinningFriction=1.0,
                             rollingFriction=0.0001,
                             # rollingFriction=1.0,
                             frictionAnchor=True)  # contactStiffness=0.0, contactDamping=0.0

        # Start thread to handle additional gripper constraints
        self.motor_joint = 1
        # self.constraints_thread = threading.Thread(target=self.step)
        # self.constraints_thread.daemon = True
        # self.constraints_thread.start()

        # copied from suction gripper
        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO(daniel) check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO(daniel) should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    # Control joint positions by enforcing hard contraints on gripper behavior
    # Set one joint as the open/close motor joint (other joints should mimic)
    def step(self):
        # while True:
        currj = [p.getJointState(self.body, i)[0]
                 for i in range(self.n_joints)]
        indj = [6, 3, 8, 5, 10]
        targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
        p.setJointMotorControlArray(self.body, indj, p.POSITION_CONTROL,
                                    targj, positionGains=np.ones(5))
        # time.sleep(0.001)

    # Close gripper fingers and check grasp success (width between fingers
    # exceeds some threshold)
    def activate(self, valid_obj=None):
        p.setJointMotorControl2(self.body, self.motor_joint,
                                p.VELOCITY_CONTROL, targetVelocity=1, force=100)
        if not self.external_contact():
            while self.moving():
                time.sleep(0.001)
        self.activated = True

    # Open gripper fingers
    def release(self):
        p.setJointMotorControl2(self.body, self.motor_joint,
                                p.VELOCITY_CONTROL, targetVelocity=-1, force=100)
        while self.moving():
            time.sleep(0.001)
        self.activated = False

    # If activated and object in gripper: check object contact
    # If activated and nothing in gripper: check gripper contact
    # If released: check proximity to surface
    def detect_contact(self):
        obj, link, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)
        else:
            return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = p.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0

    # Check grasp success
    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(p.getLinkState(self.body, 4)[0])
        rpad = np.array(p.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    # Helper functions

    def moving(self):
        v = [np.linalg.norm(p.getLinkState(
            self.body, i, computeLinkVelocity=1)[6]) for i in [3, 8]]
        return any(np.array(v) > 1e-2)

    def check_proximity(self):
        ee_pos = np.array(p.getLinkState(self.robot, self.ee)[0])
        tool_pos = np.array(p.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = p.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac
