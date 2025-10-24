import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import model.modules.StackedHourGlass as M
from model.modules.WristRefinementBranch import WristRefinementBranch, heatmap_to_coord, generate_forearm_mask      # uses new defined class

class myUpsample(nn.Module):
	 #def __init__(self):
		 #super(myUpsample, self).__init__()
		 #pass
	 def forward(self, x):
		 return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


class Hourglass(nn.Module):
    """docstring for Hourglass"""
    def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(M.Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(M.Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(M.Residual(self.nChannels,self.nChannels))

            self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(M.Residual(self.nChannels,self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        Upsampling Layer (Can we change this??????)
        As per Newell's paper upsamping recommended
        """
        self.up = myUpsample()#nn.Upsample(scale_factor = self.upSampleKernel)


    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions>1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class StackedHourGlass(nn.Module):
	"""docstring for StackedHourGlass"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints, in_ch=3):
		super(StackedHourGlass, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints

		self.start = M.BnReluConv(in_ch, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = M.Residual(64, 128)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = M.Residual(128, 128)
		self.res3 = M.Residual(128, self.nChannels)

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

		for _ in range(self.nStack):
			_hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
			_chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, x):
		x = self.start(x)
		x = self.res1(x)
		x = self.mp(x)
		x = self.res2(x)
		x = self.res3(x)
		out = []

		features_for_refine_list = []  # store features for each stack

		for i in range(self.nStack):
			x1 = self.hourglass[i](x)           # run hourglass
			x1 = self.Residual[i](x1)           # residual modules
			features_for_refine = self.lin1[i](x1)  # <- high-dimensional features (C=256)
    
			# Save features for refinement
			features_for_refine_list.append(features_for_refine)
			

			# Convert features to heatmaps
			out.append(self.chantojoints[i](features_for_refine))

			# Prepare input for next stack
			x1 = self.lin2[i](features_for_refine)
			x = x + x1 + self.jointstochan[i](out[i])

		self.last_features = features_for_refine_list[-1]
		return (out)
		
class StackedHourGlassWithWristRefine(StackedHourGlass):
     def __init__(self, nChannels, nStack, nModules, numReductions, nJoints, in_ch=3):
          super().__init__(nChannels, nStack, nModules, numReductions, nJoints, in_ch)

          # ---- Add wrist refinement branch ----
          self.wrist_refine = WristRefinementBranch(in_channels=nChannels, out_channels=1)
         
          """
          # Fusion layer: combine main + refined wrist heatmaps
          self.wrist_fusion = nn.Sequential(
               nn.Conv2d(4, 2, kernel_size=1),  # 2 main wrists + 2 refined = 2 fused
               nn.Sigmoid()                      # keep heatmaps in [0,1]
          )
          """

          # Joint indices (adjust based on your dataset)
          self.elbow_idx = 7        
          self.left_elbow_idx = 10  
          self.wrist_indices = [6, 11]  
         

     def forward(self, x):
          # Run base stacked hourglass
          out = super().forward(x)  # list of heatmaps per stack

          # Take last stack's outputs for wrist refinement
          final_heatmaps = out[-1].clone()  # (B, nJoints, H, W)
          features = self.last_features  # get the features without changing return type
          #B,C,H,W = final_heatmaps.shape
          B, nJ, H, W = final_heatmaps.shape
          device = final_heatmaps.device

          # Extract elbow heatmap
          right_elbow = final_heatmaps[:, 7:8, :, :]
          left_elbow  = final_heatmaps[:, 10:11, :, :]
          main_wrists = final_heatmaps[:, [6,11], :, :]  # (B,2,H,W)
          
          # Convert to coords
          right_elbow_coords = heatmap_to_coord(right_elbow)
          left_elbow_coords  = heatmap_to_coord(left_elbow)
          right_wrist_coords = heatmap_to_coord(main_wrists[:,0:1,:,:])
          left_wrist_coords  = heatmap_to_coord(main_wrists[:,1:2,:,:])

         # Forearm masks
          right_forearm = generate_forearm_mask(elbow_coords=right_elbow_coords, wrist_coords=right_wrist_coords, H=H, W=W, sigma=2.0)
          left_forearm  = generate_forearm_mask(elbow_coords=left_elbow_coords, wrist_coords=left_wrist_coords, H=H, W=W, sigma=2.0)



          # ---- Wrist refinement ----
          right_wrist_refined, right_forearm, right_roi = self.wrist_refine(features, right_elbow, right_forearm)   # (B,1,H_f,W_f)
          left_wrist_refined, left_forearm, left_roi  = self.wrist_refine(features, left_elbow, left_forearm)      


          
          # Upsample refined wrists to match main_wrists spatial size
          right_wrist_refined = F.interpolate(right_wrist_refined, size=(H, W), mode='bilinear', align_corners=False)
          left_wrist_refined  = F.interpolate(left_wrist_refined,  size=(H, W), mode='bilinear', align_corners=False)

          # Fuse main + refined
          #fused_wrists = self.wrist_fusion(torch.cat([main_wrists[:, 0:1], right_wrist_refined,
          #                                           main_wrists[:, 1:2], left_wrist_refined], dim=1))  # (B,2,H,W)
        
         # Confidence-guided fusion
          right_conf = torch.clamp(main_wrists[:,0:1,:,:],0,1)
          left_conf  = torch.clamp(main_wrists[:,1:2,:,:],0,1)

          fused_right = right_conf * main_wrists[:,0:1,:,:] + (1-right_conf)*right_wrist_refined
          fused_left  = left_conf  * main_wrists[:,1:2,:,:] + (1-left_conf)*left_wrist_refined

          right_wrist_backbone = main_wrists[:, 0:1, :, :]  # (B,1,H,W)
          left_wrist_backbone  = main_wrists[:, 1:2, :, :]  # (B,1,H,W)
        
          # Replace wrist channels in final heatmaps
          
          final_heatmaps[:, 6, :, :] = fused_right.squeeze(1)
          final_heatmaps[:, 11, :, :] = fused_left.squeeze(1)
          

          # Replace last stack's output with fused version
          out[-1] = final_heatmaps
          return {
             'heatmaps': out,
             'refined_wrists': {
             'right': right_wrist_refined,
             'left': left_wrist_refined
             },
             'right_forearm': right_forearm,
             'left_forearm': left_forearm,
             'right_roi': right_roi,
             'left_roi': left_roi,


         }    

     def compute_loss(self, outputs, target, target_weight, criterion, phase="1"):
          """
          Compute loss based on training phase.
          Args:
              outputs: model outputs (dict or list)
             target: ground truth heatmaps
             target_weight: visibility weights
             criterion: loss function
             phase: "1", "2a", or "2b"
         Returns:
             total_loss: scalar tensor
         """
          if isinstance(outputs, dict):
              heatmaps = outputs['heatmaps']
              refined = outputs['refined_wrists']
          else:
              heatmaps = outputs
              refined = None

          # Base loss from backbone heatmaps
          if isinstance(heatmaps, list):
              total_loss = criterion(heatmaps[0], target, target_weight)
              for output in heatmaps[1:]:
                  total_loss += criterion(output, target, target_weight)
          else:
              total_loss = criterion(heatmaps, target, target_weight)
          # Add wrist refinement loss with weighting
          refine_weight = 0.5
          # Add wrist refinement loss in phase 2a or 2b
          if phase in ["2a", "2b"] and refined is not None:
              loss_right = criterion(refined['right'], target[:, 6:7], target_weight[:, 6:7])
              loss_left = criterion(refined['left'], target[:, 11:12], target_weight[:, 11:12])
              total_loss += refine_weight * (loss_right + loss_left)
              #print(f"Refinement loss weight: {refine_weight}")

          return {
              'total_loss': total_loss,
              'loss_right': loss_right,
              'loss_left': loss_left
          }


def get_pose_net(in_ch, out_ch, refined=True, **kwargs):
	 """
	 Build pose estimation network.

	 Args:
	 in_ch: number of input channels (e.g., 3 for RGB)
	 out_ch: number of output joints
	 refined: if True, use StackedHourGlassWithWristRefine; 
                 else use original StackedHourGlass
	 Returns:
	 	 model: PyTorch nn.Module
	 """
	 if refined:
	 	 model = StackedHourGlassWithWristRefine(
	 	 	 nChannels=256,
	 	 	 nStack=2,
	 	 	 nModules=2,
	 	 	 numReductions=4,
	 	 	 nJoints=out_ch,
	 	 	 in_ch=in_ch
	 	 )
	 	 	 
	 else:
	 	 model = StackedHourGlass(
	 	 	 nChannels=256,
	 	 	 nStack=2,
	 	 	 nModules=2,
	 	 	 numReductions=4,
	 	 	 nJoints=out_ch,
	 	 	 in_ch=in_ch
	 	 )

	 return model
