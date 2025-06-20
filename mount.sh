# Mounts the GCS buckets to the local filesystem
if [ ! -d "/home/$(whoami)/patchcore-inspection/bucket" ]; then
    mkdir -p /home/$(whoami)/patchcore-inspection/bucket
fi
cd /home/$(whoami)/patchcore-inspection/bucket

# Create directories for both buckets
if [ ! -d "analysis-pipeline-data" ]; then
    mkdir -p analysis-pipeline-data
fi

if [ ! -d "web-server-bucket--analysis-pipeline-443415" ]; then
    mkdir -p web-server-bucket--analysis-pipeline-443415
fi

# Mount both buckets
gcsfuse analysis-pipeline-data analysis-pipeline-data
gcsfuse web-server-bucket--analysis-pipeline-443415 web-server-bucket--analysis-pipeline-443415

# Verify both mounts
mount_success=true

cd analysis-pipeline-data
if [ ! -d "National_Highways" ]; then
    echo "Mount failed for analysis-pipeline-data."
    mount_success=false
else
    echo "Mount successful for analysis-pipeline-data."
fi

cd ../web-server-bucket--analysis-pipeline-443415
# Check if the bucket is accessible (you may need to adjust this check based on the bucket contents)
if [ ! "$(ls -A .)" ]; then
    echo "Mount failed for web-server-bucket--analysis-pipeline-443415."
    mount_success=false
else
    echo "Mount successful for web-server-bucket--analysis-pipeline-443415."
fi

if [ "$mount_success" = true ]; then
    echo "All mounts successful."
else
    echo "Some mounts failed."
fi