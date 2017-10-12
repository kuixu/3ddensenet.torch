--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

require 'models/3DDenseConnectLayer' 

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):cuda()
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
      model.__memoryOptimized = nil
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet or opt.optMemory == 1 then
      local optnet = require 'optnet'
      local imsize = 224
      local channel = 3
      if opt.dataset == 'imagenet' then
	 imsize = 224 
	 channel = 3
      elseif opt.dataset == 'emdb' then
	 imsize = 32
	 channel = 1
      else
	 imsize = 32
	 channel = 1
      end

      -- local imsize = opt.dataset == 'imagenet' and 224 or 48
      -- local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      local sampleInput = torch.zeros(4,channel,imsize,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})

      modelname = opt.netType
      graphOpts = {
      displayProps =  {shape='ellipse',fontsize=14, style='solid'},
      nodeData = function(oldData, tensor)
        return oldData .. '\n' .. 'Size: '.. tensor:numel()
      end
      }
   end
   if opt.plot then
      generateGraph = require 'optnet.graphgen'
      g = generateGraph(model, sampleInput, graphOpts)

      filename = 'img/net_'..modelname..'_optimized'
      graph.dot(g,modelname..'_optimized', filename) 
      graph.dot(model,modelname, filename) 
      print("Net Archtecture is saved into:" .. filename)
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   -- if opt.shareGradInput then
   --   M.shareGradInput(model)
   -- end
   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput or opt.optMemory >= 2 then
      M.shareGradInput(model, opt)
      M.sharePrevOutput(model, opt)
   end

   -- Share the contiguous (concatenated) outputs of previous layers in DenseNet.
   if opt.optMemory == 3 then
      M.sharePrevOutput(model, opt)
   end

   -- Share the output of BatchNorm in bottleneck layers of DenseNet. This requires
   -- forwarding the BN layer twice at each mini-batch, but makes the memory consumption  
   -- linear (instead of quadratic) in depth
   if opt.optMemory == 4 then
      M.shareBNOutput(model, opt)
   end


   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')


      if opt.tasktype ~= 'pixel' then
          local orig = model:get(#model.modules)
          assert(torch.type(orig) == 'nn.Linear',
            'expected last layer to be fully connected')

	  model:remove(#model.modules)
      end
      if opt.tasktype == 'pixel' then
		--d
      else
          local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
          linear.bias:zero()
          model:add(linear:cuda())
      end
   end
   --print(model)


   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            require 'nngraph'
            local cudnn = require 'cudnn'
	    --require 'models/DenseConnectLayer'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:type(opt.tensorType) 
      -- old
      -- model = dpt:cuda()
   end


   local lossfunc = nn.CrossEntropyCriterion()
   if opt.lossfunc == 'nll' then 
       lossfunc = nn.ClassNLLCriterion()
       print("Criterion: ClassNLL")
   elseif opt.lossfunc == 'mse' then
       lossfunc = nn.MSECriterion()
       print("Criterion: MSECriterion")
   else
       print("Criterion: CrossEntropy")
   end

   local criterion = lossfunc

   -- if opt.netType == 'subvolumesup' then   
   --    print("subvolumesup criterion..")
   --    -- criterion = nil
   --    criterion = nn.ParallelCriterion(true)
   --    for i = 1,9 do
   --        criterion:add(nn.CrossEntropyCriterion())
   --    end
   -- end



   -- local criterion = nn.MultiLabelMarginCriterion():cuda()

   --local criterion = nn.CrossEntropyCriterion()
   --criterion = criterion_filter.Single(criterion, 1):cuda() 
   --local criterion = nn.CrossEntropyCriterion()
   -- local criterion = nn.ClassNLLCriterion():cuda()
   return model, criterion:cuda()
end

-- function M.shareGradInput(model)
--    local function sharingKey(m)
--       local key = torch.type(m)
--       if m.__shareGradInputKey then
--          key = key .. ':' .. m.__shareGradInputKey
--       end
--       return key
--    end
-- 
--    -- Share gradInput for memory efficient backprop
--    local cache = {}
--    model:apply(function(m)
--       local moduleType = torch.type(m)
--       if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
--          local key = sharingKey(m)
--          if cache[key] == nil then
--             cache[key] = torch.CudaStorage(1)
--          end
--          m.gradInput = torch.CudaTensor(cache[key], 1, 0)
--       end
--    end)
--    for i, m in ipairs(model:findModules('nn.ConcatTable')) do
--       if cache[i % 2] == nil then
--          cache[i % 2] = torch.CudaStorage(1)
--       end
--       m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
--    end
-- end

function M.shareGradInput(model, opt)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' and moduleType ~= 'nn.Concat' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
   for i, m in ipairs(model:findModules('nn.Concat')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
   print(cache)
end

function M.sharePrevOutput(model, opt)
   -- Share contiguous output for memory efficient densenet
   local buffer = nil
   model:apply(function(m)
      local moduleType = torch.type(m)
      if moduleType == 'nn.DenseConnectLayerCustom' then
         if buffer == nil then
            buffer = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.input_c = torch[opt.tensorType:match('torch.(%a+)')](buffer, 1, 0)
      end
   end)
end

function M.shareBNOutput(model, opt)
   -- Share BN.output for memory efficient densenet
   local buffer = nil
   model:apply(function(m)
      local moduleType = torch.type(m)
      if moduleType == 'nn.DenseConnectLayerCustom' then
         if buffer == nil then
            buffer = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.net1:get(1).output = torch[opt.tensorType:match('torch.(%a+)')](buffer, 1, 0)
      end
   end)
end

function M.copyModel(t, s)
   local wt, ws = t:parameters(), s:parameters()
   assert(#wt==#ws, 'Model configurations does not match the resumed model!')
   for l = 1, #wt do
      wt[l]:copy(ws[l])
   end
   local bn_t, bn_s = {}, {}
   for i, m in ipairs(s:findModules('cudnn.SpatialBatchNormalization')) do
      bn_s[i] = m
   end
   for i, m in ipairs(t:findModules('cudnn.SpatialBatchNormalization')) do
      bn_t[i] = m
   end
   assert(#bn_t==#bn_s, 'Model configurations does not match the resumed model!')
   for i = 1, #bn_s do
      bn_t[i].running_mean:copy(bn_s[i].running_mean)
      bn_t[i].running_var:copy(bn_s[i].running_var)
   end
end


return M
