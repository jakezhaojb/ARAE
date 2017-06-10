--[[ 
   Utils file
--]]

require 'nn'
require 'nngraph'

--[[ Break a string "25-30-100" into table {25,30,100} ]]--
function convert_option(s)
   local out = {}
   local args = string.split(s, '-')
   for _, x in pairs(args) do
      x = string.gsub(x, 'n', '-')
      local y = tonumber(x)
      if y == nil then
	 error("Parsing arguments: " .. s .. " is not well formed")
      end
      out[1+#out] = y
   end
   return out
end
--[[ dictionary loading ]]--
function load_dict(dictpath)
   assert(path.exists(dictpath))
   local dict, idict = {}, {}
   for line in io.lines(dictpath) do
      -- split the line
      local w, id = unpack(string.split(line, " +"))
      local id = tonumber(id)
      assert(#idict+1 == id)
      dict[w] = id
      idict[#idict+1] = w
   end
   return dict, idict
end

--[[ cast tensor to words ]]--
function convert_to_word(ts, idict)
   local str = ""
   for i = 1, ts:size(1) do
      str = str .. idict[ts[i]] .. " "
   end
   return str
end

--[[ cast words to tensor ]]--
function convert_to_idx(word, dict)
   local splits = string.split(word, " +")
   local ts = torch.Tensor(#splits+2)
   for i = 1, #splits do
      ts[i+1] = dict[splits[i]]~=nil and dict[splits[i]] or 2
   end
   ts[1] = 3
   ts[#splits+2] = 4
   return ts
end

--[[ RNN utils ]]--
function clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i =1,#paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

--[[ zeroize table ]]--
function zero_table(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

--[[ augmentation function: permutation ]]--
function permute(source)
  local new_source = source:clone()
  new_source[{{2, source:size(1)-1}}]:copy(
    source[{{2, source:size(1)-1}}]:index(1, torch.randperm(source:size(1)-2):long()))
  return new_source
end

--[[ augmentation function: deletion ]]--
function delete(source, p)
  if source:size(1) == 3 then
    return source
  end  
  local new_source = source:clone()
  local l = 2
  for t = 2, source:size(1)-1 do
    if math.random() > p then
      new_source[l]:copy(source[t])
      l = l+1
    end    
  end
  if l == 2 then
    return source
  end  
  new_source[l]:fill(4)
  return new_source[{{1, l}}]
end

--[[ augmentation function: word swapping ]]--
function swap(source, p)
  local new_source = source:clone()
  for t = 2, source:size(1)-2 do -- skip <s>, </s>    
    if math.random() < p then      
      new_source[t]:copy(source[t+1])
      new_source[t+1]:copy(source[t])
    end          
  end
  return new_source
end

