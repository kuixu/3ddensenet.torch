--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
-- multi file
dofile './provider.lua'
--local DataLoader = require 'dataloader'
local DataLoader = require 'dataloaderMulti'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

time = sys.execute('date +%Y%m%d%H%M%S') 
print(time)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)
-- print(model)
-- graph.dot(model.fg,"model")
-- Data loading
train_files = getDataFiles(opt.train_data) 
test_files = getDataFiles(opt.test_data)
-- local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- shuffle train files
   local train_file_indices = torch.randperm(#train_files)
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss
   for fn = 1, #train_files do
       opt.gendata = train_files[train_file_indices[fn]]
       time = sys.execute('date +%Y%m%d%H%M%S') 
       print('['.. time ..'] File ' .. fn .. ': loading train file:' .. opt.gendata)
       local trainLoader = DataLoader.create(opt,'train')
       trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
   end

   -- Run model on validation set
   -- shuffle test files
   local test_file_indices = torch.randperm(#test_files)
   local testTop1 =0
   local testTop5 =0
   for fn = 1, #test_files do
       opt.gendata = test_files[test_file_indices[fn]]
       time = sys.execute('date +%Y%m%d%H%M%S') 
       print('['.. time ..'] File ' .. fn .. ': loading test file:' .. opt.gendata)
       local valLoader = DataLoader.create(opt,'val')
       local tmptestTop1, tmptestTop5 = trainer:test(epoch, valLoader)
       testTop1 = testTop1 + tmptestTop1
       testTop5 = testTop5 + tmptestTop5
   end
   testTop1 = testTop1 / #test_files
   testTop5 = testTop5 / #test_files

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end

   if epoch % 10 == 0 then 
   	checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   end
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
