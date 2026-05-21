-- instance_lock_steal.lua — atomic token-CAS lock steal.
--
-- Replaces the lock key's value with a fresh token only if the stored
-- value still matches ``prev_token`` at the moment of the call. The
-- caller does the live-process check (``os.kill(pid, 0)``) before
-- invoking this script; the script's CAS guard ensures the steal is a
-- no-op if the original holder renewed in the race window between the
-- liveness check and the steal.
--
-- KEYS layout:
--   KEYS[1] = lock key      e.g. "{prefix}:lock:{name}"
--
-- ARGV layout:
--   ARGV[1] = prev_token    — token observed by the caller (must match)
--   ARGV[2] = new_token     — client-minted opaque random string
--   ARGV[3] = ttl_ms        — positive integer (PX precision)
--
-- Returns:
--   1  — lock stolen (key replaced with ``new_token`` under ``ttl_ms``)
--   0  — current value differs from ``prev_token`` (a renew or a new
--        acquire raced us; caller must re-observe and decide)
--
-- Invalid TTL / empty token surface as script errors (caller bugs).

local ttl_ms = tonumber(ARGV[3])
if not ttl_ms or ttl_ms < 1 then
    return redis.error_reply('instance_lock_steal: ttl_ms must be a positive integer, got ' .. tostring(ARGV[3]))
end
if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('instance_lock_steal: prev_token must be non-empty')
end
if ARGV[2] == nil or ARGV[2] == '' then
    return redis.error_reply('instance_lock_steal: new_token must be non-empty')
end

local current = redis.call('GET', KEYS[1])
if current ~= ARGV[1] then
    return 0
end

redis.call('SET', KEYS[1], ARGV[2], 'PX', ttl_ms)
return 1
