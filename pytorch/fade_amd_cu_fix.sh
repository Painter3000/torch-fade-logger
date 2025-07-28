#!/bin/bash
# =============================================================================
# FADE PyTorch RX 6800 XT CU Count Fix
# =============================================================================
# This script patches PyTorch to correctly report 72 Compute Units (CUs) 
# instead of AMD's broken 36 CU count for the RX 6800 XT GPU.
#
# Issue: AMD's HIP/ROCm stack incorrectly reports only 36 multiProcessorCount
# Fix: Detect RX 6800 XT and correct the value to actual 72 CUs
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TARGET_FILE="./aten/src/ATen/hip/HIPContext.cpp"
BACKUP_FILE="${TARGET_FILE}.backup"

echo -e "${BLUE}=== FADE PyTorch RX 6800 XT CU Count Fix ===${NC}"
echo -e "${BLUE}Fixing AMD's broken multiProcessorCount reporting${NC}"
echo ""

# Check if we're in PyTorch root directory
if [[ ! -f "setup.py" ]] || [[ ! -d "torch" ]] || [[ ! -f "$TARGET_FILE" ]]; then
    echo -e "${RED}❌ Error: This script must be run from PyTorch root directory${NC}"
    echo -e "${RED}   Make sure you're in the pytorch/ folder and $TARGET_FILE exists${NC}"
    exit 1
fi

echo -e "${YELLOW}📍 Target file: ${TARGET_FILE}${NC}"

# Check if file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    echo -e "${RED}❌ Error: Target file not found: $TARGET_FILE${NC}"
    exit 1
fi

# Create backup
echo -e "${BLUE}💾 Creating backup: ${BACKUP_FILE}${NC}"
cp "$TARGET_FILE" "$BACKUP_FILE"

# Check if already patched
if grep -q "FADE PATCH" "$TARGET_FILE"; then
    echo -e "${YELLOW}⚠️  File appears to already be patched!${NC}"
    echo -e "${YELLOW}   Skipping patch application.${NC}"
    echo -e "${YELLOW}   To re-patch, restore from backup first:${NC}"
    echo -e "${YELLOW}   cp ${BACKUP_FILE} ${TARGET_FILE}${NC}"
    exit 0
fi

# Verify we can find the target function
if ! grep -q "void initDeviceProperty(DeviceIndex device_index)" "$TARGET_FILE"; then
    echo -e "${RED}❌ Error: Could not find target function in $TARGET_FILE${NC}"
    echo -e "${RED}   The PyTorch version might be incompatible${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Applying FADE patch...${NC}"

# Apply the patch using sed
# We need to be careful with the multiline replacement
python3 << 'EOF'
import re

# Read the file
with open('./aten/src/ATen/hip/HIPContext.cpp', 'r') as f:
    content = f.read()

# Define the original pattern to match
original_pattern = r'''void initDeviceProperty\(DeviceIndex device_index\) \{
  hipDeviceProp_t device_prop\{\};
  AT_CUDA_CHECK\(hipGetDeviceProperties\(&device_prop, device_index\)\);
  device_properties\[device_index\] = device_prop;
\}'''

# Define the replacement
replacement = '''void initDeviceProperty(DeviceIndex device_index) {
  hipDeviceProp_t device_prop{};
  AT_CUDA_CHECK(hipGetDeviceProperties(&device_prop, device_index));
  
  // FADE PATCH: Fix AMD's broken multiProcessorCount for RX 6800 XT
  if (device_prop.multiProcessorCount == 36 && 
      strstr(device_prop.name, "RX 6800 XT") != nullptr) {
    FADE_LOG("DEVICE", "Fixed multiProcessorCount: 36 -> 72 for RX 6800 XT");
    device_prop.multiProcessorCount = 72;
  }
  
  device_properties[device_index] = device_prop;
}'''

# Apply the replacement
new_content = re.sub(original_pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

if new_content == content:
    print("❌ Pattern not found - patch may be incompatible")
    exit(1)

# Write the modified content back
with open('./aten/src/ATen/hip/HIPContext.cpp', 'w') as f:
    f.write(new_content)

print("✅ Patch applied successfully")
EOF

# Verify the patch was applied
if grep -q "FADE PATCH" "$TARGET_FILE"; then
    echo -e "${GREEN}✅ Patch applied successfully!${NC}"
    echo ""
    echo -e "${GREEN}🎯 What this patch does:${NC}"
    echo -e "${GREEN}   • Detects RX 6800 XT GPU by name${NC}"
    echo -e "${GREEN}   • Fixes multiProcessorCount: 36 → 72${NC}"
    echo -e "${GREEN}   • Logs the fix via FADE_LOG${NC}"
    echo -e "${GREEN}   • Enables full GPU utilization${NC}"
    echo ""
    echo -e "${YELLOW}📋 Next steps:${NC}"
    echo -e "${YELLOW}   1. Rebuild PyTorch: python setup.py develop${NC}"
    echo -e "${YELLOW}   2. Test: python -c \"import torch; print(torch.cuda.get_device_properties(0))\"${NC}"
    echo -e "${YELLOW}   3. Verify: multiProcessorCount should now show 72${NC}"
    echo ""
    echo -e "${BLUE}💾 Backup saved as: ${BACKUP_FILE}${NC}"
    echo -e "${BLUE}🔄 To revert: cp ${BACKUP_FILE} ${TARGET_FILE}${NC}"
else
    echo -e "${RED}❌ Patch verification failed!${NC}"
    echo -e "${RED}   Restoring backup...${NC}"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 FADE PyTorch patch completed successfully!${NC}"
echo -e "${GREEN}   Your RX 6800 XT will now report correct 72 CUs${NC}"
echo -e "${GREEN}   Performance should improve significantly!${NC}"
