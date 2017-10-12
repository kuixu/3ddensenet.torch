local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      --local m = torch.load(file, 'ascii')
      local m = torch.load(file)
      if not data then
         -- data = m.data:t()
         -- labels = m.labels:squeeze()
         data = m.data
         labels = m.labels
      else
         -- data = torch.cat(data, m.data:t(), 1)
         -- labels = torch.cat(labels, m.labels:squeeze())
         data = torch.cat(data, m.data, 1)
         labels = torch.cat(labels, m.labels,1)
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   -- labels:add(1)

   -- data = data:contiguous():view(-1, 3, 32, 32),
   return {
      data = data,
      labels = labels,
   }
end

local trainData = convertToTensor({
'/home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x3_0.t7',
'/home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x7_0.t7',
})
torch.save('/home/scs4850/kuixu/data/cryoem/insilico_res_map_3/test.t7',trainData)
-- 
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x12_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x15_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x19_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x5_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x8_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x11_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x8_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x23_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x19_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x22_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x16_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x9_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x18_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x20_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x21_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x6_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x12_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x13_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x2_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x17_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x15_1.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x10_0.t7
-- /home/scs4850/kuixu/data/cryoem/insilico_res_map_3/x22_1.t7
