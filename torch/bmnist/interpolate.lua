--[[
  Z space interpolation file
--]]

require 'cutorch' 
require 'cunn' 
require 'cudnn' 
require 'nngraph'
require 'dpnn'
paths.dofile("aux.lua")
paths.dofile("mnist.lua")

-------------- Setting ----------------
local cmd = torch.CmdLine()
cmd:option('--imgname', 'output.png', 'save image name')
cmd:option('--modelpath', '', 'model path')
cmd:option('--ninterp', 10, 'number of interpolations')
cmd:option('--batchSize', 10, 'how many images to show')
local opt_eval = cmd:parse(arg or {})

-------------- Model reloading ----------------
local loaded = torch.load(opt_eval.modelpath)
local model, opt = unpack(loaded)
local AE = model:get(1)
local AE_ = nn.Sequential():add(AE:get(1)):add(AE:get(3))
local D = model:get(2)
local G = model:get(3)
model:evaluate()

-------------- Data ----------------
dataloader = MNISTLoader(opt)
dataloader:cuda()
function binarize(x, res)
   local maxval = 0.995
   local minval = -0.995
   res:resizeAs(x):copy(x:gt((maxval+minval)/2))
   return res
end

-------------- Z-space interpolation ---------------
local noise_v = torch.CudaTensor(opt_eval.ninterp, opt.noiseDim)
local noise_l = torch.CudaTensor(opt.noiseDim)
local noise_r = torch.CudaTensor(opt.noiseDim)
--[[ interpolation on one group of left/right z vectors ]]--
local function interp()
   local line = torch.linspace(0, 1, opt_eval.ninterp)
   noise_l:normal()
   noise_r:normal()
   for i = 1, opt_eval.ninterp do
      noise_v:select(1, i):copy(noise_l*line[i] + noise_r*(1-line[i]))
   end
   local fake_hid = G:forward(noise_v)
   local fake_gen = AE_:get(2):forward(fake_hid):float()
   local _, fake_gen_max = fake_gen:max(2)
   local fake_gen_max = fake_gen_max:mul(2):add(-3)
   return fake_gen_max
end
-- calling interplation function
local todisp = {}
for i = 1, opt_eval.batchSize do
   local this_todisp = interp()
   for j = 1, opt_eval.ninterp do
      todisp[#todisp+1] = this_todisp[j]
   end  -- end for 
end  -- end for i
-- dumping out
local todisp = image.toDisplayTensor({input=todisp, nrow=opt_eval.ninterp, padding=1})
image.save(opt_eval.imgname, todisp)
