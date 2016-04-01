project = 'library'
path = strcat('./',project,'/');

fid = fopen(strcat(path,'list.txt'),'r');
name = {};
time = [];
while(1)
    line = fgets(fid);
    if(~ischar(line))
        break;
    end
    line = strsplit(line);
    name = [name; line(1)];
    time = [time; str2double(line(2))];
end
fclose(fid);

images = {};
for i = 1:size(name)
    images = [images; imread(strcat(path,'aligned_',name{i}))];
end

Zmin = 0;
Zmax = 255;
w = zeros(Zmax-Zmin+1,1);
for i = Zmin:Zmax
    if i <= (Zmin+Zmax)/2
        w(i+1) = i - Zmin;
    else
        w(i+1) = Zmax - i;
    end
end

%random get N location
N = 100;
P = size(images,1);
height = size(images{1},1);
width = size(images{1},2);

B = log(time);

HDRimage = zeros(size(images{1}));
G = [];
for color = 1:3
    Z = zeros(N,P,3);
    for i = 1:N
        x = randi(width);
        y = randi(height);
        for j = 1:P
            Z(i,j,:) = images{j}(y,x,:);
        end
    end
    [g, lE] = gsolve(Z(:,:,color), B, 10, w);
    for x = 1:width
        for y = 1:height
            denominator = 0;
            numerator = 0;
            for j = 1:P
                Zij = images{j}(y,x,color);
                denominator = denominator + w(Zij+1);
                numerator = numerator + w(Zij+1)*(g(Zij+1)-B(j));
            end
            HDRimage(y,x,color) = exp(numerator/denominator);
        end
    end
end
index = find(~isfinite(HDRimage));
HDRimage(index) = (HDRimage(i-1)+HDRimage(i+1))/2;
rgb = tonemap(HDRimage);

hdrwrite(HDRimage,strcat(path, project,'.hdr'));
imwrite(rgb,strcat(path,project,'.jpg'));