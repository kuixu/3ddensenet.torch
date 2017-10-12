--
--  Copyright (c) 2016, Kui XU.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  EMDB, ImageNet and CIFAR-10 datasets
--

require 'hdf5'

local M = {}

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)
   --local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
   -- if paths.filep(opt.gendata) then
   --    cachePath = opt.gendata
   -- elseif not paths.filep(cachePath) or not isvalid(opt, cachePath) then
   --    paths.mkdir('gen')
   --    local script = paths.dofile(opt.dataset .. '-gen.lua')
   --    script.exec(opt, cachePath)
   -- end
   -- print("using :" .. cachePath)

   local imageInfo 
   if opt.hdf then
      local script = paths.dofile(opt.datatype .. '-hdf.lua')
      imageInfo = script.loadhdf(opt, split)
   else
      local script = paths.dofile(opt.datatype .. '-t7.lua')
      imageInfo = script.loadt7(opt, split)
   --else
   --   imageInfo = torch.load(cachePath)
   end

   local Dataset = require('datasets/' .. opt.datatype)
   return Dataset(imageInfo, opt, split)
end

return M
