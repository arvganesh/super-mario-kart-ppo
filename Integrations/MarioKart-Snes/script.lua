time_elapsed = 0

function time_penalty_reward ()
    time_elapsed = time_elapsed + 1
    local reward = 0.0
    -- -- Time penalty for not moving
    if data.speed >= 50.0 then
        reward = data.speed
    else
        reward = (data.speed - 50.0) * -1
    end

    -- -- off-road penalty
    if data.surface ~= 64 then
        reward = -1000.0
    end

    return reward + -0.01 * time_elapsed
end

function basic_clipped_reward ()
    -- Off-road or Collision
    if data.surface ~= 64 or data.collision ~= 0 or data.backward == 0x10 or data.speed == 0.0 then
        return -1.0
    end

    -- Ensures speed > 0 reward get's a positive reward
    return 1.0
end

function log_speed_reward_normalized ()
    -- Off-road or Collision
    if data.surface ~= 64 or data.collision ~= 0 or data.backward == 0x10 or data.speed == 0 then
        return -1.0
    end

    -- Ensures speed of 0 gets 0 reward and increases monotonically
    return (math.log(data.speed / 900.0 + 0.001) - math.log(0.001)) / (math.log(1.001) - math.log(0.001))
end

-- function log_speed_reward ()
--     -- Off-road or Collision
--     if data.surface ~= 64 or data.collision ~= 0 or data.backward == 0x10 then
--         return -1.0
--     end
--     -- Ensures speed of 0 gets 0 reward and increases monotonically
--     return math.log(data.speed + 0.001) - math.log(0.001)
-- end

-- function simple_clipped_reward ()
--     -- Off-road or Collision
--     if data.surface ~= 64 or data.collision ~= 0 or data.backward == 0x10 or data.speed == 0 then
--         return -1.0
--     end

--     -- Ensures speed of 0 gets 0 reward and increases monotonically
--     return return 1.0
-- end