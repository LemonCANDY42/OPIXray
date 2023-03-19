import sys
from pathlib import Path

from data import OPIXray_CLASSES, OPIXrayDetection
from detection_draw import *
from test import *
from data.OPIXray import LabelType

sys.path.append("./utils")
from utils.predict_struct import result_struct
import cv2


class OPIXrayDetectionSingle(OPIXrayDetection):
	def __init__(self, img_path, *kargs, **kwargs):
		super(OPIXrayDetectionSingle, self).__init__(*kargs, **kwargs)
		self.img_path = Path(img_path)
		self.ids.append(self.img_path.absolute())


def test_net_single_img(save_folder, net, cuda, dataset, transform, top_k,
												im_size=300, thresh=0.05):
	num_images = len(dataset)
	'''
	all detections are collected into:
	all_boxes[cls][image] = N x 5 array of detections in
	(x1, y1, x2, y2, score)

	'''

	all_boxes = [[[] for _ in range(num_images)]
							 for _ in range(len(labelmap) + 1)]
	# timers
	_t = {'im_detect': Timer(), 'misc': Timer()}
	output_dir = get_output_dir('ssd300_120000', set_type)
	det_file = os.path.join(output_dir, 'detections.pkl')
	# if(k==0):
	# img = x.int().cpu().squeeze().permute(1,2,0).detach().numpy()
	# cv2.imwrite('edge_s.jpg',img)
	#    x = self.edge_conv2d(x)
	# rgb_im = rgb_im.int().cpu().squeeze().permute(1,2,0).detach().numpy()
	# cv2.imwrite('rgb_im.jpg', rgb_im)
	# for i in range(6):
	#    im = Image.fromarray(edge_detect[i]*255).convert('L')
	#    im.save(str(i)+'edge.jpg')
	# x = self.edge_conv2d.edge_conv2d(x)
	# else:
	# for i in range(num_images):
	i = 0
	im, gt, h, w, og_im = dataset.pull_item(i)
	# img = im.int().cpu().squeeze().permute(1, 2, 0).detach().numpy()
	# cv2.imwrite('/mnt/SSD/results/orgin'+str(i)+'.jpg', img)
	# im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))

	im = im.type(torch.cuda.FloatTensor)
	x = Variable(im.unsqueeze(0))

	if args.cuda:
		x = x.cuda()
	_t['im_detect'].tic()
	detections = net(x)  # .data
	detect_time = _t['im_detect'].toc(average=False)
	# skip j = 0, because it's the background class
	# //
	# //
	# print("detections:", detections.size(1))
	class_correct_scores, class_coordinate_dict = result_struct(detections, h, w, all_boxes, OPIXray_CLASSES,
																															thresh=thresh)
	print(class_correct_scores)

	return class_correct_scores, class_coordinate_dict, og_im


# with open(det_file, 'wb') as f:
#     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

# print('Evaluating detections')
# evaluate_detections(all_boxes, output_dir, dataset)

# python OPIXray/DOAM/single_img_test.py --save_image /project/data/result/1.jpg --trained_model /project/train/models/DongYingbest_loss.pth --image /project/data/1284/DYAJJ_20220727_V19_p_train_conveyor_belt_9_8190.jpg --confidence_threshold 0.1
if __name__ == '__main__':
	# EPOCHS = [45]
	# EPOCHS = [40,45,50, 55, 60, 65, 70, 75, 80,85,90,95,100,105,110,115,120,125,130,135,140,145]
	# EPOCHS = [130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255]
	# EPOCHS = [90, 95, 100, 105, 110, 115, 120, 125]
	# EPOCHS = [255]
	# print(EPOCHS)
	# for EPOCH in EPOCHS:
	reset_args()

	parser.add_argument('--image',
											default='OPIXray_Dataset/train/train_image/009069.jpg', type=str,
											help='image file path to inference')

	args = parser.parse_args()
	sys.argv = []
	# load net
	num_classes = len(labelmap) + 1  # +a1 for background
	if args.cuda:

		net = build_ssd('test', 300, num_classes)  # initialize SSD
		# net._modules['vgg'][0] = nn.Conv2d(4, 64, kernel_size=3, padding=1)
		net.load_state_dict(torch.load(args.trained_model))
		print('cuda')
	else:
		net = build_ssd('test', 300, num_classes, mode='cpu')
		# net._modules['vgg'][0] = nn.Conv2d(4, 64, kernel_size=3, padding=1)
		net.load_state_dict(torch.load(args.trained_model, map_location="cpu"))
		print('no cuda')
	net.eval()
	# print('Finished loading model!')
	# load data
	dataset = OPIXrayDetectionSingle(img_path=args.image, root=args.OPIXray_root,
																	 # BaseTransform(300, dataset_mean),
																	 target_transform=OPIXrayAnnotationTransform(), phase='test', type=LabelType.DongYing)
	if args.cuda:
		net = net.cuda()
		cudnn.benchmark = True
	# evaluation

	print(net)
	result = test_net_single_img(args.save_folder, net, args.cuda, dataset,
															 None, args.top_k, 300,
															 thresh=args.confidence_threshold)
	image = draw_with_coordinate(*result)
	im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.imwrite(args.save_image, im_rgb)
