_base_ = [
    '../models/fcn_hr18.py', '../datasets/coco-trash.py',
    '../runtime/default_runtime.py', '../schedules/schedule_20k.py'
]
classes = ("Backgroud", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
model = dict(decode_head=dict(num_classes=11))
seed = 1995
data = dict(train=dict(classes=classes),
            val=dict(classes=classes),
            test=dict(classes=classes))