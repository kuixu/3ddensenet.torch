--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local datasetname
local tasktype
local opts
local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt,split)
   -- The train and val loader
   local dataset = datasets.create(opt, split)
   local loader = M.DataLoader(dataset, opt, split)

  --  for i, split in ipairs{'train', 'val'} do
  --     local dataset = datasets.create(opt, split)
  --     loaders[i] = M.DataLoader(dataset, opt, split)
  --  end
   

   -- return table.unpack(loaders)
   return loader
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   datasetname = opt.dataset
   opts = opt
   local function init()
      require('datasets/' .. opt.datatype)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)
   print("datarun=self.batchSize:" .. self.batchSize)
   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1)
               -- print("----===---sz:".. sz .. " indices:")
               -- print(indices)
               local batch, imageSize, target, targetSize
               target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = sample.input
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
		     --if tasktype == "pixel" then
		     --    batch = torch.FloatTensor(sz,table.unpack(imageSize))
		     --    target = torch.IntTensor(sz, table.unpack(targetSize))  
                     --elseif datasetname == "emdb" then
                     --    batch = torch.FloatTensor(sz,nCrops, table.unpack(imageSize))
                     --    target = torch.IntTensor(sz,nCrops, table.unpack(imageSize))
                     --else
                     batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                     --end
                  end
                  batch[i]:copy(input)
                  --if datasetname == "emdb" then
                  --    target[i]:copy(sample.target)
                  --else
                  target[i] = sample.target
                  --end

               end
               collectgarbage()
               batch = batch:view( sz * nCrops, table.unpack(imageSize))
               -- ========== emdb data precess ============
               --if datasetname == "emdb" then
               --    target = target:view( sz * nCrops , table.unpack(targetSize))
                   -- =========== ignore label area ==========
                   -- ignore xukui edit 2017-03-03
                   -- if opts.ignore then
                   --    for _,id in ipairs(opts.ignorelabels) do  
                   --        target[target:eq(id)] = opts.nClasses + 1 
                   --    end 
                   --    -- re-label
                   --    i=1 
                   --    for _,id in ipairs(opts.relabels) do  
                   --        target[target:eq(id)]=i
                   --        i=i+1
                   --    end 
                   --    target[target:eq(opts.nClasses + 1)] = i
                   --    -- =========== ignore label area ==========
                   -- end 
               --else
               target = target
               --end
               return {
                  input = batch,
                  target = target:view(target:numel())
                  -- target = target:view(sz * nCrops , table.unpack(imageSize))
                  -- target = target:resize(sz * nCrops,target:size(2)*target:size(3)*target:size(4),1)
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
