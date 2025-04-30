import os
import sys
import random
import ctypes
import numpy as np
import moderngl
import glm
from PIL import Image


# ==============================
# 参数与目录配置
# ==============================
WIDTH, HEIGHT = 1300, 800         # 最终背景图尺寸
SAT_SCALE = 0.004                 # 卫星精灵渲染时的缩放比例

# 数据集输出目录
DATASET_DIR = "./dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LAB_DIR = os.path.join(DATASET_DIR, "labels")
TRAIN_IMG_DIR = os.path.join(IMG_DIR, "train")
VAL_IMG_DIR   = os.path.join(IMG_DIR, "val")
TRAIN_LAB_DIR = os.path.join(LAB_DIR, "train")
VAL_LAB_DIR   = os.path.join(LAB_DIR, "val")
for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LAB_DIR, VAL_LAB_DIR]:
    os.makedirs(d, exist_ok=True)

# 每组生成图片数量：情况1：50+50；情况2：每次50+50，19次共 19*100 = 1900
# 最终训练集和验证集各 1000 张图片
# （情况1：100张，情况2：1900张，总计2000张；各一半到 train/val）
NUM_CASE1_TRAIN = 50
NUM_CASE1_VAL   = 50
NUM_CASE2_ITER  = 19
NUM_CASE2_TRAIN = 50
NUM_CASE2_VAL   = 50

# ==============================
# 初始化 ModernGL 与 FBO
# ==============================
ctx = moderngl.create_standalone_context()
# 为了渲染精灵，我们使用一个足够大的离屏 FBO（这里与背景图尺寸相同即可）
SPRITE_WIDTH, SPRITE_HEIGHT = WIDTH, HEIGHT  
texture_fbo = ctx.texture((SPRITE_WIDTH, SPRITE_HEIGHT), 4)
fbo = ctx.framebuffer(color_attachments=[texture_fbo])

# ==============================
# Assimp DLL路径（请根据实际情况修改）
# ==============================
assimp_path = r"D:/pythonpro/assimp/Assimp/bin/x64"
os.environ["PATH"] += os.pathsep + assimp_path
ctypes.WinDLL(r"D:/pythonpro/assimp/Assimp/bin/x64/assimp-vc143-mt.dll")
import pyassimp
# ==============================
# 加载背景图片
# ==============================
try:
    bg_image = Image.open("D:/pythonpro/space3d/space.png").convert("RGB")
except Exception as e:
    print("背景图片加载失败:", e)
    sys.exit(1)
bg_image = bg_image.resize((WIDTH, HEIGHT))

# ==============================
# 定义卫星渲染的 Shader（保持蓝白配色）
# ==============================
sat_vertex_shader = """
#version 330
in vec3 in_position;
out vec4 v_color;
uniform mat4 model;
uniform mat4 projection;
void main() {
    gl_Position = projection * model * vec4(in_position, 1.0);
    // 卫星翼部分：X轴绝对值较大则为蓝色
    if (abs(in_position.x) > 40)
        v_color = vec4(0.6, 0.8, 1.0, 1.0);
    else
        v_color = vec4(1.0, 1.0, 1.0, 1.0);
}
"""
sat_fragment_shader = """
#version 330
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
"""
sat_prog = ctx.program(vertex_shader=sat_vertex_shader, fragment_shader=sat_fragment_shader)

# 设置透视投影矩阵（固定）
projection = glm.perspective(glm.radians(45.0), SPRITE_WIDTH/SPRITE_HEIGHT, 0.1, 100.0)
sat_prog["projection"].write(np.array(projection.to_list(), dtype="f4").tobytes())
# 固定视图矩阵：摄像机位于 (0,0,10)，观察原点
view = glm.lookAt(glm.vec3(0, 0, 10), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

# ==============================
# 加载卫星模型
# ==============================
def load_model(filename):
    print(f"加载模型: {filename}")
    try:
        with pyassimp.load(filename) as scene:
            if not scene.meshes:
                raise ValueError("模型文件中无可用网格数据")
            mesh = scene.meshes[0]
            vertices = np.array(mesh.vertices, dtype=np.float32)
            indices = np.array(mesh.faces, dtype=np.uint32).flatten()
            vbo = ctx.buffer(vertices.tobytes())
            ibo = ctx.buffer(indices.tobytes())
            vao = ctx.simple_vertex_array(sat_prog, vbo, "in_position", index_buffer=ibo)
            return vao, len(indices)
    except Exception as e:
        print("加载模型时出错:", e)
        return None, 0

satellite, num_indices = load_model("D:/pythonpro/space3d/satellite.glb")
if satellite is None:
    print("卫星模型加载失败")
    sys.exit(1)
else:
    print("卫星模型加载成功, 索引数:", num_indices)

# ==============================
# 辅助函数：渲染卫星精灵（只绘制卫星，不绘背景），清屏为透明
# ==============================
def render_satellite(model_matrix):
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 0.0)  # 清除为全透明
    sat_prog["model"].write(np.array(model_matrix.to_list(), dtype="f4").tobytes())
    satellite.render(moderngl.TRIANGLES)
    data = fbo.read(components=4, alignment=1)
    img = Image.frombytes("RGBA", (SPRITE_WIDTH, SPRITE_HEIGHT), data).transpose(Image.FLIP_TOP_BOTTOM)
    return img

# ==============================
# 核心函数：给定3D模型变换（model_matrix），渲染卫星精灵，
# 对精灵进行2D随机旋转和平移，贴入背景并计算标签
# 返回 (final_img, label_line)
# 标签格式：x_center y_center bbox_width bbox_height angle2d（均归一化，angle2d为2D旋转角度）
# ==============================
def generate_sample(model_matrix):
    # 1. 渲染卫星精灵
    sprite_full = render_satellite(model_matrix)
    sprite_np = np.array(sprite_full)
    alpha = sprite_np[:, :, 3]
    mask = alpha > 10
    if not mask.any():
        raise ValueError("卫星精灵渲染为空，请检查参数")
    ys, xs = np.where(mask)
    crop_box = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
    sprite_cropped = sprite_full.crop(crop_box)
    # 得到裁剪后精灵尺寸
    sprite_w, sprite_h = sprite_cropped.size

    # 2. 随机2D旋转（绕Z轴旋转），angle2d为标签输出
    angle2d = random.uniform(0, 360)
    rotated_sprite = sprite_cropped.rotate(angle2d, expand=True)
    rotated_w, rotated_h = rotated_sprite.size

    # 3. 在背景中随机选择粘贴位置，保证精灵完整出现在背景中
    max_x = WIDTH - rotated_w
    max_y = HEIGHT - rotated_h
    paste_x = random.randint(0, int(max_x)) if max_x > 0 else 0
    paste_y = random.randint(0, int(max_y)) if max_y > 0 else 0

    # 4. 计算旋转后精灵实际的不透明区域（利用alpha通道）
    rot_np = np.array(rotated_sprite)
    alpha_rot = rot_np[:, :, 3]
    mask_rot = alpha_rot > 10
    if not mask_rot.any():
        raise ValueError("旋转后卫星图像为空")
    ys_rot, xs_rot = np.where(mask_rot)
    # 在 rotated_sprite 坐标系中的不透明区域边界
    sprite_bbox = (int(xs_rot.min()), int(ys_rot.min()), int(xs_rot.max())+1, int(ys_rot.max())+1)
    # 计算在背景图中的绝对位置
    final_bbox = (
        paste_x + sprite_bbox[0],
        paste_y + sprite_bbox[1],
        paste_x + sprite_bbox[2],
        paste_y + sprite_bbox[3]
    )
    bbox_center_x = (final_bbox[0] + final_bbox[2]) / 2.0
    bbox_center_y = (final_bbox[1] + final_bbox[3]) / 2.0
    bbox_width = final_bbox[2] - final_bbox[0]
    bbox_height = final_bbox[3] - final_bbox[1]

    norm_center_x = bbox_center_x / WIDTH
    norm_center_y = bbox_center_y / HEIGHT
    norm_width = bbox_width / WIDTH
    norm_height = bbox_height / HEIGHT

    # label_line = f"{0} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f} {angle2d:.2f}"
    label_line = f"{0} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
    # 5. 将旋转后的卫星精灵贴入背景中
    bg_rgba = bg_image.convert("RGBA")
    final_img = bg_rgba.copy()
    final_img.paste(rotated_sprite, (paste_x, paste_y), rotated_sprite)
    final_img = final_img.convert("RGB")
    return final_img, label_line

# ==============================
# 数据集生成：
#   分为两部分：
#  (1) 情况1：无额外3D旋转（直接用 view*scale）
#  (2) 情况2：加入3D旋转：额外绕 y 轴旋转 angle3d 度，绕 x 轴旋转 2*angle3d 度
# 最终训练集和验证集各 1000 张图片，共 2000 张
# ==============================
img_counter = 0

# 基础模型矩阵：仅缩放，不旋转（情况1）
scale_mat = glm.scale(glm.mat4(1.0), glm.vec3(SAT_SCALE))
base_model = view * scale_mat

# --- 情况1 ---
print("生成情况1（无额外3D旋转）的样本...")
for split, num in [('train', NUM_CASE1_TRAIN), ('val', NUM_CASE1_VAL)]:
    for i in range(num):
        try:
            final_img, label_line = generate_sample(base_model)
        except Exception as e:
            print("生成样本出错：", e)
            continue
        if split == 'train':
            out_img_path = os.path.join(TRAIN_IMG_DIR, f"{img_counter:05d}.jpg")
            out_lab_path = os.path.join(TRAIN_LAB_DIR, f"{img_counter:05d}.txt")
        else:
            out_img_path = os.path.join(VAL_IMG_DIR, f"{img_counter:05d}.jpg")
            out_lab_path = os.path.join(VAL_LAB_DIR, f"{img_counter:05d}.txt")
        final_img.save(out_img_path)
        with open(out_lab_path, "w") as f:
            f.write(label_line + "\n")
        print(f"保存样本 {img_counter:05d}（情况1，{split}），标签: {label_line}")
        img_counter += 1

# --- 情况2 ---
print("生成情况2（加入3D旋转）的样本...")
# 对于 19 次不同的 3D 旋转，每次固定一个随机角 angle3d
for iter in range(NUM_CASE2_ITER):
    angle3d = random.uniform(0, 360)
    # 绕 y 轴旋转 angle3d 度
    rot_y = glm.rotate(glm.mat4(1.0), glm.radians(angle3d), glm.vec3(0, 1, 0))
    # 绕 x 轴旋转 2*angle3d 度
    rot_x = glm.rotate(glm.mat4(1.0), glm.radians(2 * angle3d), glm.vec3(1, 0, 0))
    extra_rot = rot_y * rot_x
    # 新的模型矩阵：先额外旋转，再缩放，再乘视图矩阵
    model_3d = view * (extra_rot * scale_mat)
    for split, num in [('train', NUM_CASE2_TRAIN), ('val', NUM_CASE2_VAL)]:
        for i in range(num):
            try:
                final_img, label_line = generate_sample(model_3d)
            except Exception as e:
                print("生成样本出错：", e)
                continue
            if split == 'train':
                out_img_path = os.path.join(TRAIN_IMG_DIR, f"{img_counter:05d}.jpg")
                out_lab_path = os.path.join(TRAIN_LAB_DIR, f"{img_counter:05d}.txt")
            else:
                out_img_path = os.path.join(VAL_IMG_DIR, f"{img_counter:05d}.jpg")
                out_lab_path = os.path.join(VAL_LAB_DIR, f"{img_counter:05d}.txt")
            final_img.save(out_img_path)
            with open(out_lab_path, "w") as f:
                f.write(label_line + "\n")
            print(f"保存样本 {img_counter:05d}（情况2，iter={iter+1}，{split}），标签: {label_line}")
            img_counter += 1

print("数据集生成完毕！")
