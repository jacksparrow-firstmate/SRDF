import torch
import os
import func_utils
from pipeline import srdf
from datasets.dataset_all import DOTA,HRSC,TEXT
import decoder
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='SRDF eval')
    parser.add_argument('--num_epoch', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=720, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=1280, help='Resized image width')
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_HRSC.pth',
                        help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='hrsc', help='Name of dataset')

    parser.add_argument('--data_dir', type=str, default='C:\\Users\\savvy\\Desktop\\datasets\\HRSC2016',
                        help='Data directory')
    parser.add_argument('--phase', type=str, default='eval', help='Phase choice= {train, test, eval}')

    args = parser.parse_args()
    return args

class EvalModule(object):
    def __init__(self, dataset, num_classes, model, decoder):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder


    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def evaluation(self, args, down_ratio):
        save_path = 'C:/Users/savvy/Desktop/'
        # save_path = 'weights_' + args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)

        func_utils.write_results(args,
                                 self.model,
                                 dsets,
                                 down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path,
                                 print_ps=True)

        if args.dataset == 'dota':
            merge_path = 'merge_'+args.dataset
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            dsets.merge_crop_image_results(result_path, merge_path)
            return None
        else:
            ap = dsets.dec_evaluation(result_path)
            return ap

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC,'text':TEXT}
    num_classes = {'dota': 15, 'hrsc': 1,'text':1 }
    heads = {'hm': num_classes[args.dataset],
             'REG': 6,
             'theta_cls': 18,
             'theta_reg': 1,
             }
    down_ratio = 4
    decoder = decoder.DecDecoder(
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    model = srdf.SRDF(heads=heads,
                      pretrained=True,
                      down_ratio=down_ratio,
                      final_kernel=1,
                      head_conv=256)

    srdf_obj = EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
    srdf_obj.evaluation(args, down_ratio=down_ratio)
