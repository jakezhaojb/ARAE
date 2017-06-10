--[[
  Model util file
--]]

paths.dofile('utils.lua')
paths.dofile('memory.lua')

--[[ make LSTM ]]--
function make_lstm(input_size, rnn_size, vocab_size, n, opt, model)
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local RnnD={opt.rnn_size,opt.rnn_size}
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    local nameL=model..'_L'..L..'_'
    -- c,h from previous timesteps
    local prev_c = inputs[L*2]
    local prev_h = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then
        local word_vecs = nn.LookupTable(vocab_size, input_size)
        word_vecs.name = 'word_vecs' .. name
        x = word_vecs(inputs[1]) -- batch_size x word_vec_size
	input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if dropout > 0 then
        x = nn.Dropout(dropout, nil, false):usePrealloc(nameL.."dropout",
                                                        {{opt.max_batch_l, input_size_L}})(x)
      end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):usePrealloc(nameL.."i2h-reuse",	  
                                                          {{opt.max_batch_l, input_size_L}},
                                                          {{opt.max_batch_l, 4 * rnn_size}})(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false):usePrealloc(nameL.."h2h-reuse",
                                                          {{opt.max_batch_l, rnn_size}},
                                                          {{opt.max_batch_l, 4 * rnn_size}})(prev_h)
    local all_input_sums = nn.CAddTable():usePrealloc(nameL.."allinput",
                             {{opt.max_batch_l, 4*rnn_size},{opt.max_batch_l, 4*rnn_size}},
                             {{opt.max_batch_l, 4 * rnn_size}})({i2h, h2h})
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2):usePrealloc(nameL.."reshapesplit",
                                                        {{opt.max_batch_l, 4, rnn_size}})
                                           (reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():usePrealloc(nameL.."G1-reuse",{RnnD})(n1)
    local forget_gate = nn.Sigmoid():usePrealloc(nameL.."G2-reuse",{RnnD})(n2)
    local out_gate = nn.Sigmoid():usePrealloc(nameL.."G3-reuse",{RnnD})(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():usePrealloc(nameL.."G4-reuse",{RnnD})(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable():usePrealloc(nameL.."G5a",{RnnD,RnnD})({
        nn.CMulTable():usePrealloc(nameL.."G5b",{RnnD,RnnD})({forget_gate, prev_c}),
        nn.CMulTable():usePrealloc(nameL.."G5c",{RnnD,RnnD})({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable():usePrealloc(nameL.."G5d",{RnnD,RnnD})
                                 ({out_gate, nn.Tanh():usePrealloc(nameL.."G6-reuse",{RnnD})(next_c)})
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  return nn.gModule(inputs, outputs)
end

--[[ make word generator ]]--
function make_word_generator(rnn_size, query_size, vocab_size)
  local model = nn.Sequential()
  model:add(nn.JoinTable(2))  
  model:add(nn.Linear(query_size+rnn_size, rnn_size))
  model:add(nn.Tanh())
  model:add(nn.Linear(opt.rnn_size, vocab_size))  
  model:add(nn.LogSoftMax())
  return model
end

--[[ make WGAN generator ]]--
function make_gan_generator(arch, gan_z, hid)
   local arch_g = convert_option(arch)
   table.insert(arch_g, 1, gan_z)
   local seq = nn.Sequential()
   for i = 1, #arch_g-1 do
      seq:add(nn.Linear(arch_g[i], arch_g[i+1]))
      seq:add(nn.BatchNormalization(arch_g[i+1]))
      seq:add(cudnn.ReLU())
   end
   seq:add(nn.Linear(arch_g[#arch_g], hid))
   return seq
end

--[[ make WGAN discriminator/critic ]]--
function make_gan_discriminator(arch, hid)
   local seq = nn.Sequential()
   local arch_d = convert_option(arch)
   seq:add(nn.Linear(hid, arch_d[1]))
   seq:add(nn.LeakyReLU(0.2))
   for i = 1, #arch_d-1 do
      seq:add(nn.Linear(arch_d[i], arch_d[i+1]))
      seq:add(nn.BatchNormalization(arch_d[i+1]))
      seq:add(nn.LeakyReLU(0.2))
   end
   seq:add(nn.Linear(arch_d[#arch_d], 1))
   seq:add(nn.Mean())
   return seq
end
