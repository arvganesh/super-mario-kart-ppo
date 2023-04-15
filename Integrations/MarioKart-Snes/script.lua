-- Helper functions to extract information from the game state
function getLap()
	-- 133 final, starts at 127
	return data.laps - 128
end

function getCheckpoint()
	local checkpoint = data.checkpoint
	local num_checkpoints = data.num_checkpoints
	local lap = getLap()
	return checkpoint + lap * num_checkpoints
end

-- Functions that calculate reward terms --

-- Return 1 if agent hits a wall >= 7 times within 10 seconds, else 0.
wallHits = 0 
wallTimer = 0
earlyStop = false 
function wall_reward() 
	wallTimer = wallTimer + 1
	local scale_factor = 0

    -- Record a collision
    if data.collision ~= 0 and data.speed < 600 then 
        -- If we haven't recorded any hits yet, reset the timer
        if wallHits == 0 then 
            wallTimer = 0
        end
        wallHits = wallHits + 1
        if wallHits >= 7 then 
            earlyStop = true
        end
        scale_factor = 1
    end

    -- Reset timer after 600 frames (10.0 seconds)
	if wallTimer > 600 then 
		wallTimer = 0
		wallHits = 0
	end

	return scale_factor
end

-- Returns 1 if the agent passes a checkpoint, else 0.
prev_checkpoint = -1
function checkpoint_reward()
    local current_cp = getCheckpoint()
    local scale_factor = 0.0

    -- Reward if we increased checkpoints
    if current_cp > prev_checkpoint then
        prev_checkpoint = current_cp
        scale_factor = 1.0
    end

    return scale_factor
end

-- Returns 1 if the agent is off-road every 30 frames, else 0.
function terrain_reward()
    local scale_factor = 0.0
    if data.surface ~= 64 and data.current_frame % 30 == 0 and data.current_frame ~= 0 then
        scale_factor = 1.0
    end
    return scale_factor
end

-- Returns 1 or 0.5 depending on the agent's speed, else 0.
function time_reward()
    local scale_factor = 0.0
    if data.current_frame % 30 == 0 and data.current_frame ~= 0 then
        scale_factor = 1.0
        if data.speed > 700.0 then
            scale_factor = 0.5
        end
    end
    return scale_factor
end

-- Returns 1.0 if the agent is going backwards, else 0.
function backwards_reward()
    local scale_factor = 0.0
    if data.backward == 0x10 and data.current_frame % 6 == 0 then
        scale_factor = 1.0
    end
    return scale_factor
end

-- Done Conditions --
function isDoneTrain()
    if getLap() >= 5 or earlyStop then
        return true
    end
    return false
end

function overall_reward()
    local wall_cf = -0.01
    local checkpoint_cf = 3.0
    local terrain_cf = -0.15
    local time_cf = -0.01
    local backwards_cf = -0.02
    local reward = wall_cf * wall_reward() + checkpoint_cf * checkpoint_reward() + terrain_cf * terrain_reward() + time_cf * time_reward() + backwards_cf * backwards_reward()
    
    if earlyStop then
        reward = -10.0
    end

    if getLap() >= 5 then
        reward = 10.0
    end

    return reward
end