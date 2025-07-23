
# Updated function call
from src.fabric_analyzer.advanced_african_fabric_pipeline import advanced_african_fabric_pipeline


results = advanced_african_fabric_pipeline(
# "src/model_avatar_gen/seun.png",  
"src/model_avatar_gen/rita.png",  
target_colors=[
    ("Purple", (93, 21, 80)),          # A deep, powerful blue often associated with royalty in West African cultures.
    ("Emerald Green", (0, 148, 115)),         # A rich, jewel-toned green that signifies wealth and nature.
    ("Amethyst Purple", (106, 13, 173)),      # A deep purple, another color historically linked to nobility.
    ("Scarlet Red", (204, 0, 0)),             # A pure, fiery red. Crucial for confirming that the previous "red turning to orange" issue is fixed.
    ("Marigold Yellow", (255, 194, 0)), 
],
load_existing_mask=True,
use_advanced_analyzer=True,  # Enable advanced analysis
save_images=True,
trim_images=True
)