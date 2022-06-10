import moderngl
from pyrr import Matrix44
from time import sleep

from base_window import BaseWindowConfig


class PhongWindow(BaseWindowConfig):
    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.sphere_obj = self.load_scene("sphere.obj")
        self.vao = self.sphere_obj.root_nodes[0].mesh.vao.instance(self.program)

    def init_shader_vairable(self, var_name):
        setattr(self, f"var_{var_name}", self.program[var_name])

    def init_shaders_variables(self):
        self.var_projection = self.program["projection"]
        self.var_view = self.program["view"]
        self.var_view_position = self.program["view_position"]
        self.var_object_color = self.program["object_color"]
        self.var_light_position = self.program["light_position"]
        self.var_light_color = self.program["light_color"]
        self.var_shininess_param = self.program["shininess_param"]
        self.var_ambient_param = self.program["ambient_param"]
        self.var_specular_param = self.program["specular_param"]
        self.var_diffuse_param = self.program["diffuse_param"]

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        view = Matrix44.look_at(
            (3.0, 1.0, -5.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )

        self.var_projection.write(projection.astype("f4"))
        self.var_view.write(view.astype("f4"))
        self.var_view_position.value = (6.0, 0.0, 0.0)

        # Light details
        self.var_light_position.value = (9.0, 1.0, -43.0)#(7.0, -5.0, -3.0)
        self.var_light_color.value = (0.5, 0.5, 0.5)

        # Material parameters
        self.var_shininess_param.value = 20.0
        self.var_ambient_param.value = 0.8
        self.var_specular_param.value = 0.6
        self.var_diffuse_param.value = 0.2

        # Object details
        self.var_object_color.value = (0.9, 0.1, 0.9)

        self.vao.render()
        sleep(frame_time)
