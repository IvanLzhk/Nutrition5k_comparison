# paths:
METADATA_PATH="data\\nutrition5k_dataset\\metadata\\dish_metadata_cafe1.csv"
IMAGERY_ROOT="data\\nutrition5k_dataset\\imagery"
TRAIN_IDS_PATH="data\\nutrition5k_dataset\\dish_ids\\splits\\rgb_train_ids.txt"
TEST_IDS_PATH="data\\nutrition5k_dataset\\dish_ids\\splits\\rgb_test_ids.txt"
MODEL_CHECKPOINT_DIR="checkpoints"
DINOV3_REPOSITORY="D:\\vs_projects\\dinov3"
DINOV3_WEIGHTS = (
    r"C:\Users\admin\.cache\huggingface\hub\models--facebook--"
    r"dinov3-vitb16-pretrain-lvd1689m\snapshots\5931719e67bbdb9737e363e781fb0c67687896bc"
)

# hyperparameters:
BATCH_SIZE=4
NUM_WORKERS=8