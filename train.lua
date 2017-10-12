--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   local dataprepro = self.opt.dataprepro
   -- print("dataprepro:" .. dataprepro)
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      -- min = torch.min(sample.input)
      max = torch.max(sample.input)
      if max <= 0 then 
          print(" | ignore this sample totally a background volume!")
          goto continue 
      end
        
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1)
      if  batchSize < self.opt.batchSize  then 
          print(" | ignore this batch which size is not enough!")
          goto continue 
      end
      local output = self.model:forward(self.input):float()
      self.target = self.target:cuda()
      -- print("train...")	
      -- print(batchSize)
      -- print(self.opt.batchSize)
      -- print(output:size())
      -- print(self.target:size())
      -- print(torch.min(self.target))
      -- print(torch.max(self.target))
      local loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, self.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Epoch: [%3d][%3d/%3d]    Time %.3f  Data %.3f  Loss %1.8f  top1 %7.3f  top5 %7.3f LR %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5, self.optimState.learningRate))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
      ::continue::
      
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      max = torch.max(sample.input)
      if max <= 0 then 
          print(" | ignore this sample totally a background volume!")
          goto continue 
      end
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      local batchSize = self.input:size(1)
      if  batchSize < self.opt.batchSize  then 
          print(" | ignore this batch which size is not enough!")
          goto continue 
      end

      local output = self.model:forward(self.input):float()

      local loss = self.criterion:forward(self.model.output, self.target)

      -- local top1, top5, log = self:computeScore(output, sample.target, nCrops)
      local top1, top5 = self:computeScore(output, self.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print((' | Test: [%3d][%3d/%3d]    Time %.3f  Data %.3f  Loss %1.8f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, loss, top1, top1Sum / N, top5, top5Sum / N) )

      timer:reset()
      dataTimer:reset()
      ::continue::
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

-- ===================================
-- ========== Compute Score ==========
-- ===================================
--
function Trainer:computeScore(output, target, nCrops)
   -- print("========================nCrops====")
   -- print(output:size())
   --print(Crops)

   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending
   -- ================= Ignore Label Area ==========
   --
   t1pred = predictions:narrow(2, 1, 1)
   -- Find which predictions match the target
   -- targ = target:long():view(batchSize, 1):expandAs(output)
   target_cp = target:clone()
   -- targ = target:resize(torch.numel(target),1)
   -- print(targ:eq(1):sum())
   -- print(targ:eq(1):sum()/batchSize*1.0)
   -- 1. calculate the count of ignore label
   -- 2. make the ignore label into another value
   local sampleCount = batchSize
   local ignoreLabelCount = 0 
   if self.opt.ignore then
      ignoreLabelCount = target:eq(self.opt.ignorelabel):sum()
      sampleCount = batchSize - ignoreLabelCount
      target_cp[target_cp:eq(self.opt.ignorelabel)]= self.opt.nClasses + 1
      -- print("27 count: " .. target_cp:eq(27):sum())
   end
   if self.opt.classweight then
      ignoreLabelCount = target:eq(1):sum()
      sampleCount = batchSize - ignoreLabelCount
      target_cp[target_cp:eq(0)]= self.opt.nClasses + 1
      -- print("27 count: " .. target_cp:eq(27):sum())
   end
   
   local correct = predictions:eq(
      target_cp:long():view(batchSize, 1):expandAs(output))
   --print("correct:")
   -- print(torch.min(correct:narrow(2, 1, 1)))
   -- print(torch.max(correct:narrow(2, 1, 1)))
   if self.opt.debug >=1 then
      t1corr = correct:narrow(2, 1, 1)
      t1label = target[t1corr:eq(1)]
      -- print(t1label:size())
      -- print(t1label)
      targetCount = "D.targ:"
      t1predCount = "D.Pred:"
      classcount =  "D.Cor2:"
      info = ""
      nclass = self.opt.nClasses
      if self.opt.ignore then
         -- info = "PredStats: " .. ignoreLabelCount .. "(ignore Label) + " .. sampleCount .. " (normal Label) = " .. batchSize .. " (total map)"
         info = "TragetStats: " .. batchSize .. " (total map) = " .. ignoreLabelCount .. " (ignore Label) + " .. sampleCount .. " (normal Label)"
         nclass = self.opt.nClasses -1
      end
      

      for i=1, nclass  do
          tmp = t1pred:eq(i):sum()
          if tmp >0 then
              t1predCount = t1predCount .. "\t" .. i .. ":" .. tmp
          end
          tmp = target:eq(i):sum()
          if tmp >0 then
              targetCount = targetCount .. "\t" .. i .. ":" .. tmp
          end
          if t1label:eq(i):sum()>0 then
             tmp = t1label[t1label:eq(i)]:numel()
             classcount = classcount .. "\t" .. i .. ":" .. tmp
          end
      end
      log = {targetCount, classcount, t1predCount, info}   
      print('')
      print(' - Classify:\t' .. self.opt.relabelstr)
      for i, line in ipairs(log) do
          print(' - ' .. line)
      end
   end
   -- Top-1 score
   -- local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / sampleCount)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   -- local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / sampleCount)

   return top1 * 100, top5 * 100 
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

-- function Trainer:copyInputs(sample)
--    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
--    -- if using DataParallelTable. The target is always copied to a CUDA tensor
--    self.input = self.input or (self.opt.nGPU == 1
--       and torch.CudaTensor()
--       or cutorch.createCudaHostTensor())
--    self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
--    self.input:resize(sample.input:size()):copy(sample.input)
--    self.target:resize(sample.target:size()):copy(sample.target)
-- end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'emdb' then
      decay = epoch >= 150 and 2 or epoch >= 100 and 1 or 0
   elseif self.opt.dataset == 'pdb' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
