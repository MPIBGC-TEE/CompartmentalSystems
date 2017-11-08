# author: Holger Metzler
# date: 2017-03-29

# set working directory to output_data folder
#setwd("PNAS/output_data_v1")
setwd("PNAS/output_data_v2")

# install and load required packages
install_and_load <- Vectorize(function(pkg){
  if (!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    if (!require(pkg, character.only = TRUE)) stop()
  }  
})

packages <- c('reshape2', 'ggplot2', 'latex2exp', 'extrafont', 'grid')
install_and_load(packages)
#font_install("fontcm")

##### plots used in the paper ####


## 1) steady state densities ##

age_density_data <- read.csv("age_dens.csv")
names(age_density_data) <- c('age', 'year', 'pool', 'value')
ss_age_density_data <- age_density_data[age_density_data$year==1765, c('age', 'pool', 'value')]

ss_stats <- ggplot(ss_age_density_data, aes(x = age, y = value, color = factor(pool))) + 
  geom_line(size = 2) +
  scale_x_continuous(TeX("Age (yr)"), expand = c(0,0)) +
  scale_y_continuous(TeX("Mass (PgC/yr)"), limits = c(0,50), expand = c(0,0)) +
  theme_bw(40) + 
  theme(#text=element_text(family="CM Roman"), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    #panel.border = element_blank(),
    panel.background = element_blank(),
    legend.position = c(0.70,0.8), 
    legend.background = element_blank(), 
    legend.title = element_blank(),
    legend.key = element_blank(),
    axis.title.y = element_text(vjust = 1.0),
    #axis.title.y = element_blank(),
    axis.title.x = element_text(vjust = 0.0),
    #axis.title.x = element_blank(),
    plot.margin = unit(c(1, 1, 0.1, 0.1), "cm")
  ) +
  scale_color_brewer(palette = "Set1", labels = list('total', 'atmosphere', 'terrestrial biosphere', 'surface ocean')) +
  guides(color=guide_legend(
    keywidth=0.5,
    keyheight=0.5,
    default.unit="inch"))
ss_stats





## 2) external data ##

# external fossil fuel input flux
input_flux_data <- read.csv('input_flux.csv')
input_flux <- input_flux_data[input_flux_data$pool==0,]

# fixed land use flux
flux_data <- read.csv('../emissions.csv', skip = 37, sep = " ")[,1:3]
names(flux_data) = c('year', 'FF', 'LU')
external_data <- melt(flux_data, id.vars = c('year'))

plt_external_data <- ggplot(external_data, aes(x = year, y = value, color = factor(variable))) + 
  geom_line(size=2) +
  scale_x_continuous("Time (yr)", limits = c(1765, 2500), expand = c(0,0), breaks = c(1765, 2000, 2250, 2500)) +
  scale_y_continuous("Flux (PgC/yr)", limits = c(0,30), expand = c(0,0), labels = c("", "10", "20", "30")) +
  theme_bw(40) + 
  theme(#text=element_text(family="CM Roman"), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank(),
        legend.position = c(0.77,0.9), 
        legend.background = element_blank(), 
        legend.title = element_blank(),
        legend.key = element_blank(), 
        #axis.title.x = element_blank(), 
        #axis.text.x = element_blank(),
        axis.title.y = element_text(vjust = 2.0),
        #axis.title.y = element_blank(),
        plot.margin = unit(c(0.5, 1.5, 0.4, 0.5), "cm"), # top, right, bottom, left
        axis.title.x = element_text(vjust = 0)
  ) +
  #scale_color_brewer(palette = "Set1", labels = list(TeX('fossil fuel input $u_A$'), TeX('land use change $f_{TA}$'))) +
  scale_color_brewer(palette = "Set1", labels = list('fossil fuel input', 'land use change')) +
  guides(color=guide_legend(
    keywidth=0.5,
    keyheight=0.5,
    default.unit="inch"))
plt_external_data


## 3) FTT densities ##
ftt_density_data <- read.csv("ftt_dens_const_ff.csv")
names(ftt_density_data) <- c('age', 'year', 'value')

years <- c(1800, 1990, 2015, 2170, 2300)
ftt_density_data <- ftt_density_data[ftt_density_data$year %in% years,]

ftt_median_const_ff <- read.csv('ftt_median_const_ff.csv')
names(ftt_median_const_ff) <- c('year', 'median')
ftt_median_const_ff <- ftt_median_const_ff[ftt_median_const_ff$year %in% years,]

# compute the density values at the median for the vertical lines
years <- ftt_median_const_ff$year
floor_medians <- floor(ftt_median_const_ff$median)
ftt_median_const_ff$density_value <- sapply(1:length(years), function(i){
  ftt_density_data[ftt_density_data$year==years[i] & ftt_density_data$age==floor_medians[i], 'value']
})

show_nr = 5
show_years = years[1:show_nr]
#labels = c("1800", "1990 (Kyoto Protocol)", "2015 (Paris Agreement)", "2170 (max. median)", "2300")[1:show_nr]
labels = c("1800", "1990 (Kyoto Protocol)", "2015 (Paris Agreement)", "2170", "2300")[1:show_nr]
plt_ftt <- ggplot(ftt_density_data[ftt_density_data$year %in% show_years,], aes(x = age, y = value, color = factor(year))) + 
  geom_line(size = 2) +
  scale_x_continuous(TeX("Age (yr)"), expand = c(0,0), limits = c(0,250)) +
  scale_y_continuous(TeX("Mass (PgC/yr)"), limits = c(0, 0.015), expand = c(0,0)) +
  theme_bw(40) + 
  theme(#text=element_text(family="CM Roman"), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    #panel.border = element_blank(),
    panel.background = element_blank(),
    legend.position = c(0.70,0.8), 
    legend.background = element_blank(), 
    legend.title = element_blank(),
    legend.key = element_blank(),
    axis.title.y = element_text(vjust = 1.0),
    #axis.title.y = element_blank(),
    axis.title.x = element_text(vjust = 0.0),
    #axis.title.x = element_blank(),
    plot.margin = unit(c(1, 1, 0.1, 0.1), "cm")
  ) +
  scale_color_brewer(palette = "Set1", labels = labels) +
  guides(color=guide_legend(
    keywidth=0.5,
    keyheight=0.5,
    default.unit="inch")) +
  geom_segment(data = ftt_median_const_ff[ftt_median_const_ff$year %in% show_years,], mapping = aes(x = median, y  =  0, xend = median, yend = density_value), linetype = 2, size = 1)
plt_ftt

