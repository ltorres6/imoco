function fm = reRatio(I, wsize)
I = mirt_dctn(I);

fm = 0;
for k = 1:wsize
    for l = 1:wsize
        for m = 1:wsize
            if k == 1 & l == 1 & m == 1
%                 disp('continuing')
                continue
            end
            fm = fm + I(k,l,m)^2;
        end
    end
end
fm = fm/(I(1,1,1)^2);
end