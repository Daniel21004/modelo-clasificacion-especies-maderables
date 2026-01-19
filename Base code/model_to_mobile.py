# Importaciones
import torch
from TriAttentionArchitectura import WoodClassifierWithTriAttention

# Ruta al modelo
model_path_attention="Modelo_E11__CON_aumento_con_tri.pt"

# Cargar el modelo
model = WoodClassifierWithTriAttention(num_classes=4, use_tri_attention=True)
state_dict = torch.load(model_path_attention, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)   
model.eval()
print("✓ Modelo cargado exitosamente")

# Seleccionar device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Guardar extensión .ptl
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

traced._save_for_lite_interpreter("modelo_android.ptl")
