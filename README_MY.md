
Por parámetros en inference():
# Solo depth (monocular depth estimation)
prediction = model.inference([image_rgb])

# Depth + pose estimation
prediction = model.inference([image_rgb], use_ray_pose=True)

# Depth + Gaussian Splatting
prediction = model.inference([image_rgb], infer_gs=True)

# Multi-view stereo (pasar múltiples imágenes)
prediction = model.inference([img1, img2, img3])
En resumen: el modelo siempre predice depth. Las otras tareas (pose, GS, sky) se habilitan/deshabilitan con flags como infer_gs, use_ray_pose, y la presencia de cam_dec/cam_enc en la config YAML.
 
Acepta cualquier resolución. El modelo redimensiona automáticamente:
- process_res=504 (por defecto): la imagen se redimensiona para que el lado más largo sea ~504px
- Las dimensiones se ajustan para ser divisibles por 14 (patch size del ViT)
