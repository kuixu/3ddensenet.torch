--
--
-- Image2D  dataset loader
-- (n,channel,width,height)

local t = require 'datasets/transforms'

local M = {}
local Image2DDataset = torch.class('conv.Image2DDataset', M)

function Image2DDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   --nsample = self.imageInfo.data:size(1)
   --nsample = nsample - nsample % opt.nGPU
   --nsample = 32
   --self.imageInfo.data = self.imageInfo.data:narrow(1,1,nsample)
   
   -- self.imageInfo.data = self.imageInfo.data * opt.dataaug
   --min = torch.min(image)
   -- print(" | min:" .. min)
   if opt.dataprepro == '1' then
      self.imageInfo.data[self.imageInfo.data:lt(0)]=0
      --min = torch.min(image)
      -- print(" | min:" .. min)
   elseif opt.dataprepro == '3' then
      -- mesh
      print("density to mesh!")
      self.imageInfo.data[self.imageInfo.data:gt(0)] = 1
   elseif opt.dataprepro == '4' then
      -- Gaussian Normalization
      print("Gaussian Normalization!")
      for i=1,self.imageInfo.data:size(1) do
          local image = self.imageInfo.data[i]
          local pre_image = image[image:gt(0)]
          pre_image = (pre_image - torch.mean(pre_image))/torch.std(pre_image)
          image[image:gt(0)] = pre_image
          self.imageInfo.data[i] = image

      end
   end
   -- self.imageInfo.labels = self.imageInfo.labels

   -- print(self.imageInfo.data:size())
   -- print("-mean/std....")
   self.split = split
end

function Image2DDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]
   return {
      input = image,
      target = label,
   }
end

function Image2DDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
local meanstd = {
   mean = {125.3, 123.0, 113.9},
   std  = {63.0,  62.1,  66.7},
}

function Image2DDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.Image2DDataset
