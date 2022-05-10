import taichi as ti
stype = input('Enter simulation type:')

if stype=="micro":
    N = 30
    R = 5
elif stype=='3body':
    N=3
    R=3
else:
    N = 5000
    R = 30
# mass = 30
radius = 0.3
dt = 0.005
G = 1
cam_speed = 0.1
paused = True

ti.init(ti.cuda)
pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
force = ti.Vector.field(3, dtype=ti.f32, shape=N)
mass = ti.field(ti.f32, shape=N)
# radius = ti.field(ti.f32, shape=N)

@ti.kernel
def compute_force(GC: ti.f32):
    for i in range(N):
        force[i] = ti.Vector([.0,.0,.0])
    for i in range(N):
        for j in range(N):
            if i != j:
                r = pos[j] - pos[i]
                force[i] += GC*mass[i]*mass[j]*r/(r.norm(1e-2)**3)

@ti.kernel
def update(dt: ti.f32):
    for i in range(N):
        vel[i] += force[i]*dt/mass[i]
        pos[i] += vel[i]*dt

@ti.kernel
def initialize():
    for i in range(N):
        k = ti.random()*2*ti.math.pi
        tpos = ti.Vector([ti.random()*2-1, ti.random()*2-1, ti.random()*2-1])
        tpos /= tpos.norm()
        tpos *= R
        pos[i] = tpos
        mass[i] = 30
        # radius[i] = 0.2

def show_options():
    global dt
    global G
    global radius
    global paused
    global cam_speed
    window.GUI.begin("Options", 0.05, 0.3, 0.25, 0.2)
    G = window.GUI.slider_float("G", G, 0, 10)
    radius = window.GUI.slider_float("Radius", radius, 0, 3)
    dt = window.GUI.slider_float("dt", dt, 0, 0.02)
    cam_speed = window.GUI.slider_float("vCamera", cam_speed, 0, 3)
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()

def render():
    camera.track_user_inputs(window, movement_speed=cam_speed, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.0, 0.8, 0.8))
    scene.particles(pos, radius=radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    canvas.scene(scene)

initialize()

window = ti.ui.Window("N body simulation!", (1080, 720))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(10.0, 0.0, 0.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(60)
canvas = window.get_canvas()


while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'p':
            paused = not paused
            print(paused)
        elif window.event.key == 'r':
            initialize()
            for i in range(N):
                force[i] = ti.Vector([.0,.0,.0])
                vel[i] = ti.Vector([.0,.0,.0])

    if not paused:
        compute_force(G)
        update(dt)
    render()
    show_options()
    window.show()