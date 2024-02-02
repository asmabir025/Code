%% Script to createmap of floats and shipboard tracks

% In order to change the elevation value from negative to positive, just
% put a minus sign in front of 'filtered_elevation' in line 38 ([Cs, Ch]..)

if(~exist('filtered_lat','var'))
    load oldbathydata.mat
end

%% 

latitudes = jsondecode(fileread('lat.json'));

longitudes = jsondecode(fileread('lon.json'));


latitudes.x2023(31) = [];
longitudes.x2023(31) = [];

%%

lat_range = [-69, -60.5];
lon_range = [-146, -82];

cmap=topo_colormap;
colormap(cmap)

figure(1);clf;
orient landscape

set(gcf, 'Position', [50, 50, 700, 600]); % Adjust the size as needed


m_proj('lambert', 'lon', lon_range, 'lat', lat_range);
%%

% Plot filled depth contours
[CS, CH] = m_contourf(filtered_lon, filtered_lat, -filtered_elevation,'edgecolor','none');
% m_pcolor(filtered_lon,filtered_lat,filtered_elevation);
shading flat


% % Add map features
m_coast('color', 'k', 'linewidth', 1);
m_grid('linestyle', 'none', 'linewidth', 2, 'tickdir', 'out');
% colorbar;

hold on;

%%

% Define an array of years and corresponding colors
years_to_plot = {'1992', '1993', '1994', '2008', '2011', '2017', '2018', '2023'};
colors_to_plot = [...
    1 0 1; ... % magenta
    0.8549 0.4392 0.8392; ... % orchid
    0.8471 0.7490 0.8471; ... % thistle
    0 1 1; ... % cyan
    0 1 0; ... % lime
    1 0.5490 0; ... % dark orange
    0.8549 0.6471 0.1255; ... % golden rod
    0 0 0; ... % black
];

years = fieldnames(latitudes);
year_list = {};
hs = NaN(size(years_to_plot));
for i = 1:numel(years)
    original_year = years{i};
    year = original_year(2:end); % Remove the first character ('x')

    if strcmp(year, '2023') % Skip the year 2023 in this loop
        continue;
   
    end
    year_list = [year_list, {year}];
    idx = find(strcmp(years_to_plot, year));
    if year == '2023'
        marker_val = 'x';
        point_val = 5;
        point_state = 0;
    else
        marker_val = 'o';
        point_val = 5;
        point_state = 1;
    end

    
    latitude_val = latitudes.(original_year);
    longitude_val = longitudes.(original_year);
    
    
    hs(i)=m_scatter(longitude_val, latitude_val, point_val, 'Marker', marker_val,'MarkerFaceAlpha', point_state,...
        'LineWidth', 1, 'MarkerFaceColor',colors_to_plot(idx,:),'MarkerEdgeColor',colors_to_plot(idx,:));
   
    hold on;
end

%%

% Separate code block for 2023 data
original_year = 'x2023';
year = '2023';
marker_val = '+';
point_val = 5;
point_state = 0;
idx = find(strcmp(years_to_plot, year));
latitude_val = latitudes.(original_year);
longitude_val = longitudes.(original_year);

hs(end + 1) = m_scatter(longitude_val, latitude_val, point_val, 'Marker', marker_val,'MarkerFaceAlpha', point_state,...
    'LineWidth', 2, 'MarkerFaceColor',colors_to_plot(idx,:),'MarkerEdgeColor',colors_to_plot(idx,:));
year_list = [year_list, {year}];

hs = hs(~isnan(hs));

% Convert year_list to numerical values for sorting
year_values = cellfun(@str2double, year_list);

% Get the sorted order based on the year_values
[~, sorted_order] = sort(year_values);

% Reorder the handles and labels according to the sorted order
sorted_hs = hs(sorted_order);
sorted_year_list = year_list(sorted_order);

% Create a new legend with the sorted handles and labels
lgd = legend(sorted_hs, sorted_year_list);
set(lgd, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 17);


cb = colorbar('northoutside'); % Put the colorbar above the plot
set(cb, 'Orientation', 'horizontal'); % Make the colorbar horizontal
ylabel(cb, 'Bathymetry (m)', 'FontSize', 15); % Add the label


% Add title and labels
% title('Depth Contour Map');
xlabel('Longitude', 'FontSize',25);
ylabel('Latitude', 'FontSize',25);

% exportgraphics(gcf, 'plottest1.pdf', 'ContentType', 'image', 'BackgroundColor', 'white');

set(gcf, 'PaperPositionMode', 'auto'); 
set(gcf, 'Color', 'white'); % Set the figure background color to none
set(gca, 'Color', 'white'); % Set the axes background color to none
set(gcf, 'InvertHardcopy', 'off'); % Ensure that background is preserved
print(gcf, '-dpdf', 'bathy_plot.pdf'); % Print the figure to a PDF file



