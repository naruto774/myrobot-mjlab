import mujoco

model = mujoco.MjModel.from_xml_path(
  "src/mjlab/asset_zoo/robots/marsdog/xmls/assets/marsdog.xml"
)
data = mujoco.MjData(model)

# 1. 约束是否激活
print("eq_active:", data.eq_active)

# 2. 关节 id
q_tarsus = model.joint("rr_tarsus_joint").qposadr[0]
q_calf = model.joint("rr_calf_joint").qposadr[0]

# 固定 base 或关掉 gravity 后再测
model.opt.gravity[:] = 0
data.qpos[q_calf] = 0.5
data.qpos[q_tarsus] = 0.0
mujoco.mj_forward(model, data)

for _ in range(500):
  mujoco.mj_step(model, data)
  err = data.qpos[q_tarsus] - data.qpos[q_calf]
  if _ % 50 == 0:
    print(f"err={err:.6f}")
