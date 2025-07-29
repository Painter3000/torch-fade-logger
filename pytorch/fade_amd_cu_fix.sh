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

# Detect actual CU count from rocminfo
echo -e "${BLUE}🔍 Detecting actual CU count from rocminfo...${NC}"
CU_COUNT=$(rocminfo | grep -A5 "Name: gfx" | grep "Compute Unit" | head -n1 | grep -o '[0-9]\+')

if [ -z "$CU_COUNT" ]; then
    echo -e "${YELLOW}⚠️  Could not detect CU count from rocminfo, using static RX 6800 XT fix (72 CUs)${NC}"
    USE_STATIC_FIX=true
else
    echo -e "${GREEN}✅ Detected ${CU_COUNT} Compute Units from hardware${NC}"
    USE_STATIC_FIX=false
fi

# Choose patch method based on user preference and detection
echo -e "${BLUE}🛠️  Choose patch method:${NC}"
echo -e "${BLUE}   1) Dynamic fix (auto-detect CU count from rocminfo)${NC}"
echo -e "${BLUE}   2) Static fix (hardcoded RX 6800 XT: 36→72)${NC}"
echo -e "${BLUE}   3) Wrapper method (override hipGetDeviceProperties)${NC}"

read -p "Select method [1-3]: " PATCH_METHOD

case $PATCH_METHOD in
    1)
        if [ "$USE_STATIC_FIX" = true ]; then
            echo -e "${RED}❌ Cannot use dynamic fix without rocminfo detection${NC}"
            exit 1
        fi
        echo -e "${BLUE}🔧 Applying dynamic CU fix (${CU_COUNT} CUs)...${NC}"
        ;;
    2)
        echo -e "${BLUE}🔧 Applying static RX 6800 XT fix (72 CUs)...${NC}"
        ;;
    3)
        echo -e "${BLUE}🔧 Applying wrapper method...${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid selection${NC}"
        exit 1
        ;;
esac

# Pass environment variables to Python
export PATCH_METHOD
export CU_COUNT

# Apply the patch using Python
python3 << 'EOF'
import re
import os

# Read the file
with open('./aten/src/ATen/hip/HIPContext.cpp', 'r') as f:
    content = f.read()

# Get patch method from environment
patch_method = os.environ.get('PATCH_METHOD', '2')
cu_count = os.environ.get('CU_COUNT', '72')

if patch_method == '1':
    # Dynamic fix using rocminfo-detected CU count
    original_pattern = r'''void initDeviceProperty\(DeviceIndex device_index\) \{
  hipDeviceProp_t device_prop\{\};
  AT_CUDA_CHECK\(hipGetDeviceProperties\(&device_prop, device_index\)\);
  device_properties\[device_index\] = device_prop;
\}'''

    replacement = f'''void initDeviceProperty(DeviceIndex device_index) {{
  hipDeviceProp_t device_prop{{}};
  AT_CUDA_CHECK(hipGetDeviceProperties(&device_prop, device_index));
  
  // FADE PATCH: Dynamic CU fix based on rocminfo detection
  // Detected hardware CU count: {cu_count}
  if (device_prop.multiProcessorCount < {cu_count}) {{
    FADE_LOG("DEVICE", "Fixed multiProcessorCount: %d -> {cu_count} (auto-detected)", device_prop.multiProcessorCount);
    device_prop.multiProcessorCount = {cu_count};
  }}
  
  device_properties[device_index] = device_prop;
}}'''

elif patch_method == '2':
    # Static RX 6800 XT fix
    original_pattern = r'''void initDeviceProperty\(DeviceIndex device_index\) \{
  hipDeviceProp_t device_prop\{\};
  AT_CUDA_CHECK\(hipGetDeviceProperties\(&device_prop, device_index\)\);
  device_properties\[device_index\] = device_prop;
\}'''

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

elif patch_method == '3':
    # Wrapper method - add at top of file
    include_pattern = r'(#include "c10/hip/fade_logger\.h")'
    wrapper_code = '''
// FADE PATCH: Wrapper function for hipGetDeviceProperties
static hipError_t FADE_hipGetDeviceProperties(hipDeviceProp_t* prop, int device_id) {
    hipError_t status = hipGetDeviceProperties(prop, device_id);
    if (status == hipSuccess) {
        // Auto-fix known broken CU counts
        if (prop->multiProcessorCount == 36 && strstr(prop->name, "RX 6800 XT") != nullptr) {
            FADE_LOG("DEVICE", "FADE Wrapper: Fixed multiProcessorCount 36 -> 72 for RX 6800 XT");
            prop->multiProcessorCount = 72;
        }
        // Add more GPU fixes here as needed
    }
    return status;
}

// Override hipGetDeviceProperties with our wrapper
#define hipGetDeviceProperties FADE_hipGetDeviceProperties

'''
    
    # Add wrapper after fade_logger include
    new_content = re.sub(include_pattern, r'\1' + wrapper_code, content)
    
    if new_content == content:
        print("❌ Could not find include pattern for wrapper method")
        exit(1)
    
    # Write the modified content back
    with open('./aten/src/ATen/hip/HIPContext.cpp', 'w') as f:
        f.write(new_content)
    
    print("✅ Wrapper method applied successfully")
    exit(0)

# Apply the replacement for methods 1 and 2
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
