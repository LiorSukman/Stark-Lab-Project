function diff_spike = discrite_derivate( spike )

diff_spike = [spike(2:end),0] - spike;
diff_spike = diff_spike(1:end-1);

end

