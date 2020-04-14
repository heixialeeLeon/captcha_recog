alphabets = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#image_folder = '/data/vott-csv-export'
#image_folder = '/data/captcha/航华验证码'
# train_folder = '/data/captcha/huahang/train'
#train_folder = '/home/peizhao/data/captcha/train_total'

train_folder = '/data/captcha/huahang/train'
test_folder = '/data/captcha/huahang/test'

# mashiji data set
# train_folder = '/home/peizhao/data/captcha/train'
# test_folder = '/home/peizhao/data/captcha/test'

# peizhao generator dataset
# train_folder = '/home/peizhao/data/captcha/peizhao_generator/train'
# test_folder = '/home/peizhao/data/captcha/peizhao_generator/test'
train_test_rate = 0.9
image_input_size = (150,53)

#resume_model =''
resume_model ='checkpoint/captcha_epoch_50.pth'
#resume_model ='checkpoint_huahang/huahang_v3.pth'
#resume_model ='checkpoint_mashiji/mashiji_resnet18_v0.pth'

#trainning parameters
epoch = 350
lr = 0.0001
batch_size = 32
step = [50, 100, 150]
display_interval = 20
save_per_epoch = 10