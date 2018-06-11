--[[
  auxiliary functions
--]]

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

--[[ Table deepcopy ]]--
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

--[[ Checking tensor containing Nan ]]--
function check_nan(t)
   return t:sum()~=t:sum()
end
