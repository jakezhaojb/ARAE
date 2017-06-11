--[[
  Generate with vector interpolation
--]]

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
paths.dofile('data.lua')
paths.dofile('models.lua')
paths.dofile('utils.lua')

---------------------------------------
-------------- Setting ----------------
---------------------------------------
cmd = torch.CmdLine()
cmd:option('-gpuid', 1, [[device id]])
cmd:option('-model_file', '', [[Path to the model]])
cmd:option('-batch_size', 100, [[Number of sents in each generated batch]])
cmd:option('-batches', 10, [[Number of generated batches]])
cmd:option('-output_file', '', [[Path to the output files]])
cmd:option('-gen_length', 15, [[Generated sent max length]])
cmd:option('-mode', 'argmax', [[argmax | sample]])
opt_gen = cmd:parse(arg)
cutorch.setDevice(opt_gen.gpuid)

--[[ overwrite convert_to_word function handling <\s> ]]--
function convert_to_word(ts, idict)
   local t = {}
   for i = 1, ts:size(1) do
      if ts[i] == 4 then break end
      table.insert(t, idict[ts[i]])
   end  -- end for i = 1, 
   if opt.reverse == nil or opt.reverse ~= 1 then    
      return table.concat(t, ' ')
   else
      local u = {}
      for i = #t, 1, -1 do
        table.insert(u, t[i])
      end  -- end for i = 1, #t
      return table.concat(u, ' ')
   end  -- end if opt.reverse  
end

---------------------------------------
----------- Model loading -------------
---------------------------------------
if not paths.filep(opt_gen.model_file) then
   error("model file not found: " .. opt_gen.model_file)
end
local loaded = torch.load(opt_gen.model_file)
local modelae, param_ae, gparam_ae = unpack(loaded[1])
local modelgan, param_g, gparam_g, param_d, gparam_d = unpack(loaded[2])
opt = loaded[3]
-- CUDAisze models
modelae:cuda()
modelgan:cuda()
-- unroll RNN autoencoders
local encoder = modelae:get(1)
local decoder = modelae:get(2)
local wordgen = modelae:get(3)
wordgen:evaluate()
transferU = modelae:get(4)
if opt.init_dec == 1 then
   transferL = modelae:get(5)
end
-- unroll WGAN
local gangen = modelgan:get(1)
local gandisc = modelgan:get(2)
gangen:evaluate()
opt.max_batch_l = opt_gen.batch_size

---------------------------------------
------------- Inference ---------------
---------------------------------------
-- buffer initialization
local pad_proto = torch.ones(opt.max_batch_l):long():cuda()  -- CudaLongTensor
local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size):cuda()
local init_layer = {}
for L = 1, opt.num_layers do
   table.insert(init_layer, h_init:clone())
   table.insert(init_layer, h_init:clone())
end
local h_init_query = torch.zeros(opt.max_batch_l, opt.query_size):cuda()
local init_layer_query = {}
for L = 1, opt.num_layers_query do
   table.insert(init_layer_query, h_init_query:clone())
   table.insert(init_layer_query, h_init_query:clone())
end
local noise_v = torch.CudaTensor()
local noise_l = torch.CudaTensor()
local noise_r = torch.CudaTensor()
local genexpo = torch.LongTensor()
--[[ reset init_layer or init_layer_query  ]]--
local function reset_state(state, batch_l, t)
   if t == nil then
      local u = {}
      for i = 1, #state do
    state[i]:zero()
    table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
   else
      local u = {[t] = {}}
      for i = 1, #state do
    state[i]:zero()
    table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
   end
end
--[[ generate text function ]]--
local function gentext(n, l, mode)
   -- load dictionary
   if idict==nil then
      _, idict = load_dict(opt.dict_file)  -- path to dict
   end
   -- tensor get-ready
   local pad = pad_proto[{{1, n}}]
   noise_v:resize(n, opt.gan_z):normal()
   genexpo:resize(l, n):zero()
   -- WGAN generator fwd
   local fake_context = gangen:forward(noise_v)
   --[[ generate samples inline function ]]--
   local function generate_sample(context, result)
      local rnn_state_dec = reset_state(init_layer_query, n, 0)
      if opt.init_dec == 1 then
         rnn_state_dec[0][#rnn_state_dec[0]]:copy(transferL:forward(context))
      end
      local decoder_inputs = {}
      local pred_argmax
      -- decoder fwd
      for t = 1, l do
         decoder:evaluate()
         if t == 1 or opt.teacher_forcing == 0 then
 	    decoder_inputs[t] = {pad, table.unpack(rnn_state_dec[t-1])}
         else
 	    decoder_inputs[t] = {pred_argmax[{{}, 1}], table.unpack(rnn_state_dec[t-1])}
         end	-- end if t == 1
         rnn_state_dec[t] = decoder:forward(decoder_inputs[t])
         local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
         local pred = wordgen:forward(pred_input)
         -- greedy decoding or sampling
         if mode == 'argmax' then
       _, pred_argmax = pred:max(2)
         elseif mode == 'sample' then
       pred_argmax = torch.multinomial(pred:exp(), 1)
         end	-- end if mode == 'argmax'
         result:select(1,t):copy(pred_argmax)
       end  -- end for t = 1, l
       return result
   end  -- end local function generate_sample
   -- run the generate_sample function on the generated code
   genexpo = generate_sample(fake_context, genexpo)
   -- displaying genexpo
   for i = 1, n do
      local this_sample = genexpo:select(2,i)
      local this_str = convert_to_word(this_sample, idict)
      if output_text then
         output_text:write(this_str .. '\n')
         output_vector:write(table.concat(noise_v[i]:totable(), ' ') .. '\n')
      end  -- end if output_text
      print(this_str)
   end  -- end for i = 1
   collectgarbage()
end  -- end local function gentext

function interp(n, l, mode)
   -- load dictionary
   if idict==nil then
      _, idict = load_dict(opt.dict_file)  -- path to dict
   end
   -- tensor get-ready
   local pad = pad_proto[{{1, n}}]
   noise_v:resize(n, opt.gan_z):normal()
   genexpo:resize(l, n):zero()
   -- interpolation in the z space
   local line = torch.linspace(0, 1, n)
   noise_l:resize(opt.gan_z):normal()
   noise_r:resize(opt.gan_z):normal()
   for i = 1, n do
      noise_v:select(1, i):copy(noise_l*line[i] + noise_r*(1-line[i]))
   end
   -- WGAN generator fwd
   local fake_context = gangen:forward(noise_v)
   --[[ generate samples inline function ]]--
   local function generate_sample(context, result)
      local rnn_state_dec = reset_state(init_layer_query, n, 0)
      if opt.init_dec == 1 then
         rnn_state_dec[0][#rnn_state_dec[0]]:copy(transferL:forward(context))
      end
      local decoder_inputs = {}
      local pred_argmax
      -- decoder fwd
      for t = 1, l do
         decoder:evaluate()
         if t == 1 or opt.teacher_forcing == 0 then
 	    decoder_inputs[t] = {pad, table.unpack(rnn_state_dec[t-1])}
         else
 	    decoder_inputs[t] = {pred_argmax[{{}, 1}], table.unpack(rnn_state_dec[t-1])}
         end	-- end if t == 1
         rnn_state_dec[t] = decoder:forward(decoder_inputs[t])
         local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
         local pred = wordgen:forward(pred_input)
         -- greedy decoding or sampling
         if mode == 'argmax' then
       _, pred_argmax = pred:max(2)
         elseif mode == 'sample' then
       pred_argmax = torch.multinomial(pred:exp(), 1)
         end	-- end if mode == 'argmax'
         result:select(1,t):copy(pred_argmax)
       end  -- end for t = 1, l
       return result
   end  -- end local function generate_sample
   -- run the generate_sample function on the interpolated code
   genexpo = generate_sample(fake_context, genexpo)
   -- displaying genexpo
   for i = 1, n do
      local this_sample = genexpo:select(2,i)
      local this_str = convert_to_word(this_sample, idict)
      if output_text then
         output_text:write(this_str .. '\n')
         output_vector:write(table.concat(noise_v[i]:totable(), ' ') .. '\n')
      end  -- end if output_text
      print(this_str)
   end  -- end for i = 1, 
   print(" ")
   if output_text then
      output_text:write('\n')
      output_vector:write('\n')
   end  -- end if output_text
   collectgarbage()
end

--[[ main function ]]--
local function main()
   -- output file handle
   if string.len(opt_gen.output_file) ~= 0 then
      output_text = io.open(opt_gen.output_file .. '-text.out', 'w')
      output_vector = io.open(opt_gen.output_file .. '-vec.out', 'w')
   end
   -- generate loop
   for i = 1, opt_gen.batches do
      interp(8, opt_gen.gen_length, opt_gen.mode)
   end  -- end for i = 1, 
   -- close file handle
   if string.len(opt_gen.output_file) ~= 0 then
      output_text:close()
      output_vector:close()
   end
end  -- end local function main

main()
