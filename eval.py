def computeArea(bb):
	return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
	ibb = [max(bb1[0], bb2[0]), \
		max(bb1[1], bb2[1]), \
		min(bb1[2], bb2[2]), \
		min(bb1[3], bb2[3])]
	iArea = computeArea(ibb)
	uArea = computeArea(bb1) + computeArea(bb2) - iArea
	return (iArea + 0.0) / uArea

def computeOverlap(detBBs, gtBBs):
	aIoU = computeIoU(detBBs[0, :], gtBBs[0, :])
	bIoU = computeIoU(detBBs[1, :], gtBBs[1, :])
	return min(aIoU, bIoU)


def eval_recall(args):
	f = open(args.det_file, "r")
	dets, det_bboxes = cp.load(f)
	f.close()
	f = open(args.gt_file, "r")
	all_gts, all_gt_bboxes = cp.load(f)
	f.close()
	num_img = len(dets)
	tp = []
	fp = []
	score = []
	total_num_gts = 0
	for i in xrange(num_img):
		gts = all_gts[i]
		gt_bboxes = all_gt_bboxes[i]
		num_gts = gts.shape[0]
		total_num_gts += num_gts
		gt_detected = np.zeros(num_gts)
		if isinstance(dets[i], np.ndarray) and dets[i].shape[0] > 0:
			det_score = np.log(dets[i][:, 0]) + np.log(dets[i][:, 1]) + np.log(dets[i][:, 2])
			inds = np.argsort(det_score)[::-1]
			if args.num_dets > 0 and args.num_dets < len(inds):
				inds = inds[:args.num_dets]
			top_dets = dets[i][inds, 3:]
			top_scores = det_score[inds]
			top_det_bboxes = det_bboxes[i][inds, :]
			num_dets = len(inds)
			for j in xrange(num_dets):
				ov_max = 0
				arg_max = -1
				for k in xrange(num_gts):
					if gt_detected[k] == 0 and top_dets[j, 0] == gts[k, 0] and top_dets[j, 1] == gts[k, 1] and top_dets[j, 2] == gts[k, 2]:
						ov = computeOverlap(top_det_bboxes[j, :, :], gt_bboxes[k, :, :])
						if ov >= args.ov_thresh and ov > ov_max:
							ov_max = ov
							arg_max = k
				if arg_max != -1:
					gt_detected[arg_max] = 1
					tp.append(1)
					fp.append(0)
				else:
					tp.append(0)
					fp.append(1)
				score.append(top_scores[j])
	score = np.array(score)
	tp = np.array(tp)
	fp = np.array(fp)
	inds = np.argsort(score)
	inds = inds[::-1]
	tp = tp[inds]
	fp = fp[inds]
	tp = np.cumsum(tp)
	fp = np.cumsum(fp)
	recall = (tp + 0.0) / total_num_gts
	top_recall = recall[-1]
	print top_recall