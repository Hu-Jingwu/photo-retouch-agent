# PixCake 人像美化 panel → our face tools

Read from the user's local 像素蛋糕 (PixCake 2), 2026-11-22. Default values shown are
PixCake's own defaults for the 女 (female) audience. Columns: what we implement in
`agent_tools/face.py` and how. PixCake numbers are its own scale; ours map 0..100 to
deliberately conservative maxima, so our "100" is never extreme.

## 祛除瑕疵 Blemishes

| PixCake | default | ours | how |
|---|---|---|---|
| 祛斑祛痘 / 祛痘 / 祛斑 | 100 | `acne_spots` | small dark round spots on face skin -> inpaint |
| 质感保留 | 0 | `texture_keep` | how much pore texture survives smoothing |
| 祛黑眼圈 | 0 | `dark_circles` | under-eye region lifted to cheek lightness / colour |
| 祛眼袋 | 0 | `eye_bags` | under-eye region smoothed (edge-aware) |
| 祛油光 (脸部) | 0 | `oil_shine` | specular highlights on face skin pulled to local mean |
| 祛法令纹 (左/右) | 0 | `nasolabial` | crease band between nose wing and mouth corner filled |
| 祛痣, 祛汗珠, 祛鼻孔瑕疵, 眉纹修整, 祛双下巴, 下颌线增强, 祛胡须, 祛颈纹, 抬头纹/川字纹/眼周纹/鼻背纹/脸颊纹/木偶纹/嘴周纹, 身体祛瑕疵, 祛副乳, 祛妊娠纹, AI祛纹身, AI祛体毛 | 0 | - | not implemented (need generative inpainting / dedicated detectors) |

## 皮肤调整 Skin

| PixCake | default | ours | how |
|---|---|---|---|
| 质感磨皮 | 60 | `smoothing` | guided-filter frequency separation, pores put back |
| 水润磨皮 (脸/身) | 30 | (folded into `smoothing`) | |
| 中性灰平整 / 立体, 高低频磨皮 | 0 | - | dodge & burn style, not implemented |
| 肤色透亮 | 20 | `luminous` | light lift + slightly lower contrast on skin |
| 肤色美白 | 0 | `whitening` | lift + less yellow on skin (kept small: never changes skin tone) |
| 皮肤红润 | 0 | `rosy` | warm-pink on the cheeks |
| AI肤色统一 (脸 40 / 身 20) | 40 | `tone_unify` | low-frequency colour of skin evened out |
| AI补光, 柔化阴影 | - | main agent `subject_light` relight | |
| 面部丰盈, 皮肤质感 (哑光/细腻/水光), 光泽/纹理/清晰度, 肤色选择 | - | - | not implemented |

## 面部重塑 Face reshape (liquify by landmarks)

| PixCake | default | ours | how |
|---|---|---|---|
| 瘦脸 (左/右) | 25 | `face_slim` [L,R] | cheeks/jaw pulled to the face centre line |
| 小脸 | 20 | `small_face` | whole face scaled slightly toward its centre |
| V脸 (左/右) | 0 | `v_face` | lower jaw toward the chin |
| 颧骨 (左/右) | 0 | `cheekbone` | cheekbones inward |
| 下巴长短 | 0 | `chin_length` (±) | chin moved down / up |
| 下颌 (左/右) | 0 | `jaw` | jaw angle inward |
| 眼睛大小 (左/右) | 25 | `eye_size` (±, [L,R]) | bulge/pinch around each iris |
| 鼻子 大小 / 鼻翼 | 0 | `nose_size` (±), `nose_wings` | pinch around the nose / alar band |
| 太阳穴, 短脸, 脸宽, 发际线, 额头, 人中, 中庭, 下庭, 尖下巴, 收下巴, 3D塑颜, 眉毛 (7), 眼睛 (眼高/眼宽/眼距/眼角/倾斜/位置), 鼻高/鼻梁/鼻尖, 嘴巴 (8), 面部提拉, 脸型预设 | - | - | not implemented yet |

## 眼睛增强 / 牙齿美化

| PixCake | default | ours | how |
|---|---|---|---|
| 亮眼 | 40 | `eye_brighten` | eye region lift |
| 眼瞳 | 70 | `iris` | iris local contrast + a little saturation |
| 眼白 | 30 | `eye_whites` | sclera less red/yellow, a little lighter (capped) |
| 牙齿美白 (明度 70 / 祛黄 70) | 40 | `teeth_whiten` | teeth less yellow, a little lighter (capped, never paper-white) |
| 眼睛高光, 祛红血丝, 眼白净化, 祛红眼, 闭眼修复, 眼镜祛反光, 双眼皮, 牙齿排齐, 露龈调整 | - | - | not implemented |

## PixCake 写真 one-click preset (人像 part), read on DSC_6390

Applied from 预设 → 写真 → 写真 (tagged 人像). Colour panel: profile 质感, white balance as shot
(5438 K / tint 7), all 影调 sliders 0, no filter: the preset is almost entirely portrait work.

| Group | 写真 values | ours at 100% (`pixcake_xiezhen_100`) | ours at 70% (`pixcake_xiezhen_70`) |
|---|---|---|---|
| 祛斑祛痘 / 质感保留 | 100 / 100 | acne_spots 100, texture_keep 70 | acne_spots 70, texture_keep 70 |
| 祛油光 | 30 | oil_shine 30 | 21 |
| 祛黑眼圈 | 75 | dark_circles 75 | 52.5 |
| 祛双下巴 (程度 90, 阴影 80) | 70 | - (not implemented) | - |
| 祛颈纹 / 祛眼周纹 | 100 / 40 | - | - |
| 祛法令纹 左/右 | 50 / 50 | nasolabial 50 | 35 |
| 身体祛瑕疵 | 70 | - (main retouch handles body skin) | - |
| 质感磨皮 / 水润磨皮(身体) | 70 / 25 | smoothing 70 | 49 |
| 肤色透亮 / 美白 | 20 / 10 | luminous 20, whitening 10 | 14, 7 |
| AI肤色统一 脸 / 身 | 50 / 40 | tone_unify 50 | 35 |
| 瘦脸 左/右 | 20 / 20 | face_slim 20 | 14 |
| 下颌 左/右 | -40 / -40 | jaw 40 | 28 |
| 小脸 | 20 | small_face 20 | 14 |
| 眼睛大小 左/右, 眼高 | 40 / 40, 15 | eye_size 40 (眼高: -) | 28 |
| 鼻子 大小 / 鼻梁 / 鼻翼 | -35 / -15 / -30 | nose_size 35, nose_wings 30 (鼻梁: -) | 24.5, 21 |
| 嘴巴 大小 | -30 | mouth_size 30 | 21 |
| 亮眼 / 眼瞳 / 眼白 / 眼睛高光 | 40 / 70 / 30 / 40 | eye_brighten 40, iris 70, eye_whites 30 | 28, 49, 21 |
| 牙齿美白 (明度 70, 祛黄 70) | 40 | teeth_whiten 40 | 28 |
| 眉毛, 3D塑颜, 面部提拉, 其他脸型 | 0 | | |

### Calibration against PixCake's real export

PixCake's export of DSC_6390 with 写真 was measured with face landmarks (all lengths relative
to the distance between the pupils): eye area x1.071, nose width x0.974, mouth width x0.994,
jaw width x0.982. `evals/calibrate_pixcake.py` tuned our warp strengths (`face.GAIN`) until our
100% preset matches (eye x1.069, nose x0.973, jaw x0.984; mouth x0.985, narrowed mostly by
small_face). Our original hand-set strengths were about 2-4x too strong. At 70% we get eye
x1.048, nose x0.981, jaw x0.989, mouth x0.989.

The owner finds the full 写真 look too strong (网红脸) and prefers about 70%: that is
`pixcake_xiezhen_70`, the face sub-agent's choice for posed 写真 / beauty portraits.
Texture retention is NOT scaled down: lowering everything else must never make skin
more plastic.

Other PixCake groups seen but out of scope here: 表情管理, 妆容调整, 头发调整, 手部美化,
脚部美化, 全身美型, audience tabs (女/男/儿童/长辈女/长辈男/所有人).
