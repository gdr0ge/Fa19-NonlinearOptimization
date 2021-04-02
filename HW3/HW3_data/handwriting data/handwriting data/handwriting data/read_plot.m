% READ the training .txt file for processing
% The data matrix A created by this file has the dimension 3 *7291, where
% the first row is the correct label, the second and the third rows are the features. 


fileID = fopen('features_train.txt','r');       %open file
formatSpec = '%f %f %f';                            %specifying the reading format
sizeA=[3 inf];                                                 %specifying the size of the data matrix
A = fscanf(fileID,formatSpec,sizeA);        % reading the data matrix
fclose(fileID);

%getting the size of the matrix A
[a b]=size(A);



%plot, here I plot the distribution of digit 1 and the rest, just as what
%we have shown during lecture 7. 

figure
for i=1:b
    if A(1, i) ==1
        plot(A(2,i), A(3,i), 'b*');
    else
        plot(A(2,i), A(3,i), 'r.');
    end
    hold on;
end
xlabel('Intensity')
ylabel('Symmetry')