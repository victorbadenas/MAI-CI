function [] = save4Dcsv(mat, folder, labels1, labels2)
% save4Dcsv(allAccuracies,'',["Train" "Val" "Test"],["80-10-10" "40-20-40" "10-10-80"])
    mat = permute(mat, [3, 4, 1, 2]);
    [~, ~, dim3, dim4] = size(mat);
    
    fprintf("saving %i csv files",dim3*dim4);
    for j=1:dim4
        for i=1:dim3
            filename = sprintf("%s_%s.csv",labels2(j),labels1(i))
            redMat = squeeze(mat(:,:,i,j))
            csvwrite(folder + "/" + filename,redMat)
        end
    end
end

