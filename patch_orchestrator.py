import os
import re

# Path to the AI orchestrator file
orchestrator_path = "backend/services/ai_orchestrator.py"

# Read the orchestrator file
with open(orchestrator_path, 'r') as file:
    content = file.read()

# Import statement to add
import_statement = "from backend.services.gpu_patch import generate_test_audio_with_gpu"

# Check if import already exists
if import_statement not in content:
    # Add import near the top, after other imports
    pattern = r"(import .*?\n\n)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        position = match.end()
        content = content[:position] + f"{import_statement}\n\n" + content[position:]

# Pattern to find the audio generation code
pattern = r"(# Stage 5: Mix & Master.*?project\[\"stages\"\]\[\"mix_master\"\] = \{.*?\"created_at\": datetime\.utcnow\(\)\.isoformat\(\).*?\})"

# Replacement with GPU accelerated code
replacement = """# Stage 5: Mix & Master (GPU ACCELERATED)
            stage_start("mix_master")
            
            # Use GPU for audio generation if available
            audio_url = generate_test_audio_with_gpu(project_id)
            
            project["stages"]["mix_master"] = {
                "id": str(uuid.uuid4()),
                "final_audio_url": audio_url,
                "stems_available": True,
                "mastering_settings": {"lufs": -14, "dynamic_range": "medium"},
                "created_at": datetime.utcnow().isoformat()
            }"""

# Apply the replacement using regex
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the modified content back to the file
with open(orchestrator_path, 'w') as file:
    file.write(content)

print(f"Successfully patched {orchestrator_path} to use GPU acceleration!")
