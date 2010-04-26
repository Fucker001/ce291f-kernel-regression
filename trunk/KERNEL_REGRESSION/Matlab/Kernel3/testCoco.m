clear
clc
coco = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19];
sizeinf=size(coco,2);
for i=1:sizeinf
    new_index = floor(rand*sizeinf);
    if new_index==0
        new_index=1;
    end
    temp=coco(:,i);
    coco(:,i)=coco(:,new_index);
    coco(:,new_index)=temp;
end
coco