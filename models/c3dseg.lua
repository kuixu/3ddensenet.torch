-- 
--  Copyright (c) 2016-present, Kui XU, Tsinghua Univ.
--  All rights reserved.
-- 
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
 
require 'nn'
-- require 'cunn'
require 'cudnn'
require 'nnlr'

-- input 1
-- num of kernel 21
-- kernel size 5x5x5
F=3
-- stride size 1x1x1 
S=1
-- pad  size   2x2x2
P=1

-- make sure: (W-F+2P)/S+1
-- so: S=1, F = 2P + 1


local function createModel(opt)
      C = opt.nClasses --11
   local cfg = {64, 'M1', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}

   local features = nn.Sequential()
   do
      local iChannels = 1;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            -- features:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
            features:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
         elseif v == 'M1' then
            features:add(nn.VolumetricMaxPooling(1,1,1,1,1,1):ceil())
         else
            local oChannels = v;
            features:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1)
              :learningRate('weight',1)
              :learningRate('bias',2)
              :weightDecay('weight',1)
              :weightDecay('bias',0))
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   --features:get(1).gradInput = nil
    
   local classifier = nn.Sequential()
   --classifier:add(nn.View(2,3))
   classifier:add(nn.VolumetricConvolution(512,C,1,1,1,1,1,1)
              :learningRate('weight',1)
              :learningRate('bias',2)
              :weightDecay('weight',1)
              :weightDecay('bias',0))
   classifier:add(nn.ReLU(true))
   -- classifier:add(nn.VolumetricFullConvolution(C,C,3,22,22,1,15,15)
   classifier:add(nn.VolumetricFullConvolution(C,C,16,16,16,16,16,16)
              :learningRate('weight',1)
              :learningRate('bias',2)
              :weightDecay('weight',1)
              :weightDecay('bias',0))
   --classifier:add(nn.View(2,3))
   --classifier:add(nn.Transpose({2,3},{3,4},{4,5})) -- output is now: b,t,w,h,C
   classifier:add(nn.View(-1,C):setNumInputDims(5)) -- output is now 2D [b*t*w*h C]
   
   
   --classifier:add(nn.Dropout(0.5))
   --classifier:add(nn.Linear(4096, 4096):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0))
   --classifier:add(nn.ReLU(true))
   --classifier:add(nn.Dropout(0.5))
   --classifier:add(nn.Linear(4096, 101):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0)) --487 Sport1m-- UCF-101
   -- classifier:add(nn.LogSoftMax())

   c3dModel = nn.Sequential()
   c3dModel:add(features):add(classifier) 
   c3dModel:cuda()
   return c3dModel


    -- local net = nn.Sequential()



    -- local function Conv(inN,outN)
    --     net:add(nn.VolumetricConvolution(inN,outN,F,F,F,S,S,S,P,P,P))
    --     net:add(nn.LeakyReLU(0.1,true))
    --     net:add(nn.VolumetricDropout(0.2))
    --     return net
    -- end
    -- local function DeConv(inN,outN)
    --     net:add(nn.VolumetricFullConvolution(inN,outN,2,2,2,2,2,2))
    --     return net
    -- end
    -- local function MaxPool()
    --     net:add(nn.VolumetricMaxPooling(2,2,2,2,2,2))
    --     return net
    -- end
    -- 
    -- 
    -- 
    -- Conv(1, 32)
    -- Conv(32, 64)
    -- MaxPool()
    -- Conv(64, 128)
    -- -- Conv(128, 128)
    -- DeConv(128, 128)
    -- MaxPool()
    -- Conv(128, 256)
    -- -- Conv(128, 128)
    -- -- net:add(nn.VolumetricMaxPooling(2,2,2,2,2,2))
    -- DeConv(256,256)
    -- -- Conv(32, 64)
    -- net:add(nn.View(256))
    -- inN, outN  = 256, opt.nClasses
    -- net:add(nn.Linear(inN,outN))
    -- net:add(nn.ReLU(true))
    -- -- net:add(nn.Dropout(0.4))
    -- --if opt.lossfunc == 'nll' then
    -- --   net:add(nn.LogSoftMax())
    -- --end
    -- net:cuda()
    -- return net

end

return createModel
