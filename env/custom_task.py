import numpy as np
from panda_gym.envs.tasks.pick_and_place import PickAndPlace


class CustomHouseholdTask(PickAndPlace):
    """A household room environment for a mobile manipulator.

    The scene contains:
      - A large tiled floor
      - Four walls enclosing a ~4 m × 4 m room
      - A kitchen counter, a dining table, shelving
      - Scattered household items (cups, plates, cereal boxes, bottles)
      - The pick-and-place object and target
    """

    # Room dimensions
    ROOM_HALF = 2.0        # half-width of the room
    WALL_HEIGHT = 0.6
    WALL_THICKNESS = 0.02

    def __init__(
        self,
        sim,
        reward_type="dense",
        distance_threshold=0.05,
        goal_xy_range=1.5,
        goal_z_range=0.2,
        obj_xy_range=1.5,
    ):
        super().__init__(
            sim,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            goal_xy_range=goal_xy_range,
            goal_z_range=goal_z_range,
            obj_xy_range=obj_xy_range,
        )
        # Force safe spawning ranges to avoid colliding with furniture
        safe_range = 1.0
        self.goal_range_low = np.array([-safe_range / 2, -safe_range / 2, 0])
        self.goal_range_high = np.array([safe_range / 2, safe_range / 2, goal_z_range])
        self.obj_range_low = np.array([-safe_range / 2, -safe_range / 2, 0])
        self.obj_range_high = np.array([safe_range / 2, safe_range / 2, 0])

    # ------------------------------------------------------------------ #
    #  Scene builder                                                       #
    # ------------------------------------------------------------------ #
    def _create_scene(self) -> None:
        R = self.ROOM_HALF
        WH = self.WALL_HEIGHT
        WT = self.WALL_THICKNESS

        # ---- Floor ----
        self.sim.create_box(
            body_name="floor",
            half_extents=np.array([R + WT, R + WT, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, -0.01]),
            rgba_color=np.array([0.82, 0.78, 0.72, 1.0]),  # warm beige tile
            specular_color=np.array([0.3, 0.3, 0.3]),
        )

        # ---- Walls (4 sides) ----
        wall_color = np.array([0.92, 0.91, 0.88, 1.0])  # off-white
        # North wall (+Y)
        self.sim.create_box(
            body_name="wall_north",
            half_extents=np.array([R + WT, WT, WH / 2]),
            mass=0.0,
            position=np.array([0.0, R, WH / 2]),
            rgba_color=wall_color,
        )
        # South wall (-Y)
        self.sim.create_box(
            body_name="wall_south",
            half_extents=np.array([R + WT, WT, WH / 2]),
            mass=0.0,
            position=np.array([0.0, -R, WH / 2]),
            rgba_color=wall_color,
        )
        # East wall (+X)
        self.sim.create_box(
            body_name="wall_east",
            half_extents=np.array([WT, R + WT, WH / 2]),
            mass=0.0,
            position=np.array([R, 0.0, WH / 2]),
            rgba_color=wall_color,
        )
        # West wall (-X)
        self.sim.create_box(
            body_name="wall_west",
            half_extents=np.array([WT, R + WT, WH / 2]),
            mass=0.0,
            position=np.array([-R, 0.0, WH / 2]),
            rgba_color=wall_color,
        )

        # ---- Kitchen counter (against north wall) ----
        counter_h = 0.35
        self.sim.create_box(
            body_name="kitchen_counter",
            half_extents=np.array([0.8, 0.25, counter_h / 2]),
            mass=0.0,
            position=np.array([0.0, R - 0.27, counter_h / 2]),
            rgba_color=np.array([0.35, 0.25, 0.18, 1.0]),  # dark wood
            specular_color=np.array([0.15, 0.15, 0.15]),
        )

        # ---- Dining table (center-right) ----
        table_h = 0.30
        self.sim.create_box(
            body_name="dining_table",
            half_extents=np.array([0.40, 0.30, table_h / 2]),
            mass=0.0,
            position=np.array([1.0, -0.5, table_h / 2]),
            rgba_color=np.array([0.55, 0.35, 0.20, 1.0]),  # medium wood
            specular_color=np.array([0.1, 0.1, 0.1]),
        )
        # Table legs (4)
        leg_ext = np.array([0.02, 0.02, table_h / 2])
        leg_color = np.array([0.45, 0.30, 0.18, 1.0])
        for i, (lx, ly) in enumerate([
            (1.0 - 0.35, -0.5 - 0.25),
            (1.0 + 0.35, -0.5 - 0.25),
            (1.0 - 0.35, -0.5 + 0.25),
            (1.0 + 0.35, -0.5 + 0.25),
        ]):
            self.sim.create_box(
                body_name=f"table_leg_{i}",
                half_extents=leg_ext,
                mass=0.0,
                position=np.array([lx, ly, table_h / 2]),
                rgba_color=leg_color,
            )

        # ---- Shelf unit (against west wall) ----
        shelf_h = 0.45
        self.sim.create_box(
            body_name="shelf",
            half_extents=np.array([0.15, 0.50, shelf_h / 2]),
            mass=0.0,
            position=np.array([-R + 0.17, 0.0, shelf_h / 2]),
            rgba_color=np.array([0.70, 0.65, 0.55, 1.0]),  # light wood
        )
        # Shelf middle board
        self.sim.create_box(
            body_name="shelf_board",
            half_extents=np.array([0.14, 0.49, 0.008]),
            mass=0.0,
            position=np.array([-R + 0.17, 0.0, shelf_h * 0.55]),
            rgba_color=np.array([0.65, 0.60, 0.50, 1.0]),
        )

        # ---- Pick-and-place object (Invisible dummy for gym internals) ----
        self.sim.create_box(
            body_name="object",
            half_extents=np.array([0.01, 0.01, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, -1.0]),
            rgba_color=np.array([0, 0, 0, 0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([0.01, 0.01, 0.01]),
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, -1.0]),
            rgba_color=np.array([0, 0, 0, 0]),
        )

        # ---- Household clutter (all interactable) ----
        # Cereal box on counter (sized for Panda gripper: max ~8cm opening)
        self.sim.create_box(
            body_name="cereal_box",
            half_extents=np.array([0.025, 0.015, 0.05]),
            mass=0.05,
            position=np.array([-0.4, R - 0.27, counter_h + 0.05]),
            rgba_color=np.array([0.85, 0.25, 0.15, 1.0]),  # red box
        )

        # Set high friction on all graspable objects
        for obj_name in ["cereal_box"]:
            obj_id = self.sim._bodies_idx[obj_name]
            self.sim.physics_client.changeDynamics(
                obj_id, -1, lateralFriction=1.5, spinningFriction=0.5,
            )
        # Small rug (flat box)
        self.sim.create_box(
            body_name="rug",
            half_extents=np.array([0.35, 0.25, 0.003]),
            mass=0.0,
            ghost=True,
            position=np.array([-0.5, -1.0, 0.003]),
            rgba_color=np.array([0.15, 0.45, 0.85, 1.0]),  # bright blue rug
        )

        # ── New interactive objects ──────────────────────────

        # Book on the floor (blue, graspable)
        self.sim.create_box(
            body_name="book",
            half_extents=np.array([0.06, 0.04, 0.015]),  # flat book shape
            mass=0.05,
            position=np.array([0.3, 0.3, 0.015]),  # lying on floor
            rgba_color=np.array([0.15, 0.25, 0.75, 1.0]),  # blue
        )
        book_id = self.sim._bodies_idx["book"]
        self.sim.physics_client.changeDynamics(
            book_id, -1, lateralFriction=1.5, spinningFriction=0.5,
        )

        # Trash on the floor (grey crumpled paper)
        self.sim.create_box(
            body_name="trash",
            half_extents=np.array([0.02, 0.02, 0.02]),  # small cube
            mass=0.01,
            position=np.array([-0.6, -0.5, 0.02]),  # lying on floor
            rgba_color=np.array([0.5, 0.5, 0.5, 1.0]),  # grey
        )
        trash_id = self.sim._bodies_idx["trash"]
        self.sim.physics_client.changeDynamics(
            trash_id, -1, lateralFriction=1.5, spinningFriction=0.5,
        )

        # Dustbin in the corner (static, dark grey cylinder)
        self.sim.create_cylinder(
            body_name="dustbin",
            radius=0.12,
            height=0.40,
            mass=0.0,
            position=np.array([R - 0.35, R - 0.35, 0.20]),  # NE corner
            rgba_color=np.array([0.6, 0.6, 0.6, 1.0]),  # light grey
        )

        # ── Smart Home Devices ──────────────────────────────

        # TV on the south wall (flat screen, starts OFF = black)
        self.sim.create_box(
            body_name="tv",
            half_extents=np.array([0.35, 0.01, 0.20]),  # wide, thin, tall
            mass=0.0,
            position=np.array([0.0, -R + 0.05, 0.40]),  # mounted on south wall
            rgba_color=np.array([0.08, 0.08, 0.08, 1.0]),  # black (OFF)
        )

        # Lamp near dining table (pole + bulb)
        # Pole
        self.sim.create_cylinder(
            body_name="lamp_pole",
            radius=0.015,
            height=0.35,
            mass=0.0,
            position=np.array([1.5, -0.8, 0.175]),
            rgba_color=np.array([0.3, 0.3, 0.3, 1.0]),  # grey metal
        )
        # Bulb (starts OFF = dark grey)
        self.sim.create_cylinder(
            body_name="lamp",
            radius=0.06,
            height=0.08,
            mass=0.0,
            position=np.array([1.5, -0.8, 0.39]),  # on top of pole
            rgba_color=np.array([0.3, 0.3, 0.25, 1.0]),  # dark (OFF)
        )

        # ── Cleanable objects ───────────────────────────────

        # Dirt patch on the floor (dark brown, very thin)
        self.sim.create_cylinder(
            body_name="dirt",
            radius=0.15,
            height=0.002,
            mass=0.0,
            position=np.array([-0.2, 0.6, 0.001]),  # on the floor
            rgba_color=np.array([0.30, 0.18, 0.08, 0.9]),  # dark brown
        )

        # ── Apply Textures ───────────────────────────────
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "assets")
        
        try:
            pc = self.sim.physics_client
            
            # Load environment textures
            wood_tex = pc.loadTexture(os.path.join(assets_dir, "wood_texture.png"))
            floor_tex = pc.loadTexture(os.path.join(assets_dir, "floor_tile.png"))
            wall_tex = pc.loadTexture(os.path.join(assets_dir, "wallpaper.png"))
            
            # Load object textures
            cereal_tex = pc.loadTexture(os.path.join(assets_dir, "cereal_box.png"))
            book_tex = pc.loadTexture(os.path.join(assets_dir, "book_cover.png"))
            bin_tex = pc.loadTexture(os.path.join(assets_dir, "metal_bin.png"))
            trash_tex = pc.loadTexture(os.path.join(assets_dir, "trash_paper.png"))
            dirt_tex = pc.loadTexture(os.path.join(assets_dir, "dirt_spill.png"))
            
            # Apply Environment
            floor_id = self.sim._bodies_idx.get("floor")
            if floor_id is not None:
                pc.changeVisualShape(floor_id, -1, textureUniqueId=floor_tex)
            
            for w in ["wall_north", "wall_south", "wall_east", "wall_west"]:
                w_id = self.sim._bodies_idx.get(w)
                if w_id is not None:
                    pc.changeVisualShape(w_id, -1, textureUniqueId=wall_tex)
                
            for f in ["kitchen_counter", "dining_table", "dining_leg1", "dining_leg2", "dining_leg3", "dining_leg4", "shelf_board", "shelf_leg1", "shelf_leg2"]:
                f_id = self.sim._bodies_idx.get(f)
                if f_id is not None:
                    pc.changeVisualShape(f_id, -1, textureUniqueId=wood_tex)
                    
            # Apply Objects
            obj_map = {
                "cereal_box": cereal_tex,
                "book": book_tex,
                "trash": trash_tex,
                "dirt": dirt_tex
            }
            for obj_name, tex_id in obj_map.items():
                b_id = self.sim._bodies_idx.get(obj_name)
                if b_id is not None:
                    pc.changeVisualShape(b_id, -1, textureUniqueId=tex_id)
                
        except Exception as e:
            print(f"  [!] Warning: Could not load textures: {e}")
