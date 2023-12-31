import moderngl
from pyrr import Matrix44, Vector3

from base_window import BaseWindowConfig


class RobotWindow(BaseWindowConfig):

    def __init__(self, **kwargs):
        super(RobotWindow, self).__init__(**kwargs)

    def model_load(self):
        # downloading objects
        self.cube_obj = self.load_scene("cube.obj")
        self.sphere_obj = self.load_scene("sphere.obj")

        # added - shortcuts for rendering commands
        self.cube_link = self.cube_obj.root_nodes[0].mesh.vao.instance(self.program)
        self.sphere_link = self.sphere_obj.root_nodes[0].mesh.vao.instance(self.program)


    def init_shaders_variables(self):
        self.var_color = self.program["color"]
        self.var_pvm = self.program["pvm"]


    def render(self, time: float, frame_time: float):
        self.ctx.clear(0.8, 0.8, 0.8, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (-20.0, -15.0, 5.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )

        # Head
        self.var_color.value = (0.9, 0.0, 0.9) # Set the color to pink
        move = Matrix44.from_translation((0.0, 0.0, 5.0)) # Apply translation bu (0, 0, 5)
        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.sphere_link.render()

        # Torso
        self.var_color.value = (0.05, 0.8, 0.05) # Set the color to green
        move = Matrix44.from_translation((0.0, 0.0, 2.0)) # Apply translation by (0, 0, 2)
        move *= Matrix44.from_scale((1.0, 1.0, 2.0))
        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.cube_link.render()

        # Arms
        ## Right
        self.var_color.value = (0.2, 0.2, 0.2) # Set the color to gray
        move = Matrix44.from_translation((0.0, -3.0, 3.0))
        move *= Matrix44.from_x_rotation(3.14 / 4.0)
        move *= Matrix44.from_scale((0.5, 0.5, 1.25))
        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.cube_link.render()

        ## Left
        self.var_color.value = (0.2, 0.2, 0.2)
        move = Matrix44.from_translation((0.0, 3.0, 3.0))
        move *= Matrix44.from_x_rotation(3.14 / -4.0)
        move *= Matrix44.from_scale((0.5, 0.5, 1.25))
        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.cube_link.render()

        # Legs
        ## Right
        self.var_color.value = (1, 0.1, 0.0)
        move = Matrix44.from_translation((0.0, -2.0, -1.5))
        move *= Matrix44.from_x_rotation(3.14 / 6.0)
        move *= Matrix44.from_scale((0.5, 0.5, 1.75))
        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.cube_link.render()

        ## Left
        self.var_color.value = (1, 0.1, 0.0)
        move = Matrix44.from_translation((0.0, 2.0, -1.5))
        move *= Matrix44.from_x_rotation(3.14 / -6.0)
        move *= Matrix44.from_scale((0.5, 0.5, 1.75))

        self.var_pvm.write((projection * lookat * move).astype("f4"))
        self.cube_link.render()
