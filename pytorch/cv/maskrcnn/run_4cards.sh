# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
export NGPUS=4
export MUSA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 9527 tools/train_net.py \
--config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"  \
SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 SOLVER.MAX_ITER 180000 \
SOLVER.STEPS "(120000, 160000)" SOLVER.BASE_LR 0.01 \
MODEL.WEIGHT /path/to/e2e_mask_rcnn_R_50_FPN_1x.pth
