# author: Holger Metzler
# date: 2017-03-29

# set working directory to output_data folder
setwd("nonl_gcm_3p/output_data")

# install and load required packages
install_and_load <- Vectorize(function(pkg){
  if (!require(pkg, character.only = TRUE)){
    install.packages(pkg, char)
    if (!require(pkg, character.only = TRUE)) stop()
  }  
})

packages <- c('reshape2', 'rgl', 'plyr')
install_and_load(packages)

####### function definitions ############

# plot an age density
plot_age_density <- function(density, palette, alpha, age_stride = 1, time_stride = 1, add = FALSE){
  times <- unique(density$time)
  strided_times <- times[seq(1, length(times), age_stride)]
  if (strided_times[[length(strided_times)]] != times[[length(times)]]) strided_times <- append(strided_times, times[[length(times)]])
  
  ages <- unique(density$age)
  strided_ages <- ages[seq(1, length(ages), time_stride)]
  if (strided_ages[[length(strided_ages)]] != ages[[length(ages)]]) strided_ages <- append(strided_ages, ages[[length(ages)]])

  density <- density[density$time %in% strided_times & density$age %in% strided_ages,]

  ages <- unique(density$age)
  times <- unique(density$time)
  values = acast(density, formula = age ~ time)
  
  position <- outer(ages, times, function(a,b) a-b)
  nbcol <- length(times)
  jet.colors <- palette
  colors <- jet.colors(nbcol)
  
  # colors along the age-time diagonal
  poscol <- cut(position, nbcol)
  persp3d(ages, times, values, col = colors[poscol], alpha = alpha, add = add)
  #surface3d(ages, times, values, col = colors[poscol], alpha = alpha, add = add)
  
  density
}

find_z_value_for_age_and_time <- function(strided_density, ma, mt, age_stride, time_stride){
  if (is.nan(ma)) return(NaN)

# this code is faster, but makes stupid spikes  
#  if ((time_stride==1) && (age_stride==1)){
#    strided_density <- strided_density[strided_density$time==mt,]
#    ages <- unique(strided_density$age)
#    lower_a <- tail(ages[ages<=ma], 1)
#    upper_a <- head(ages[ages>=ma], 1)
#    vals <- strided_density[strided_density$age %in% c(lower_a, upper_a), 'value']
#    return(max(vals))
#  }
  
  times <- unique(strided_density$time)
  ages <- unique(strided_density$age)
  
  lower_t <- tail(times[times<=mt], 1)
  upper_t <- head(times[times>=mt], 1)
  
  val_between <- function(t){
    lower_age <- tail(ages[ages<=ma], 1)
    upper_age <- head(ages[ages>=ma], 1)
    
    # mean outside the plotting range?
    if (is.null(lower_age) | (is.null(upper_age))) return(NaN)
    
    lower_value <- tail(strided_density[strided_density$time==t & strided_density$age<=ma, 'value'], 1)
    upper_value <- head(strided_density[strided_density$time==t & strided_density$age>=ma, 'value'], 1)
    
    if ((length(lower_value) == 0) | (length(upper_value) == 0)) return(NaN)
    
    if (upper_age != lower_age){
      value <- lower_value + (upper_value-lower_value)*(ma-lower_age)/(upper_age-lower_age)
    } else value <- lower_value
    value
  }
  
  lower_value <- val_between(lower_t)
  upper_value <- val_between(upper_t)
  if (is.nan(lower_value) | is.nan(upper_value)) return(NaN)
  
  if (upper_t != lower_t){
    value <- lower_value + (upper_value-lower_value)*(mt-lower_t)/(upper_t-lower_t)
  } else value <- lower_value
  
  value
}

set_aspect_ratio <- function(density){
  nr_times = length(unique(density$time))
  nr_ages = length(unique(density$age))
  total = nr_ages + nr_times
  aspect3d(nr_ages/total, nr_times/total,1)
}

plot_object <- function(title, density, mean = NULL, median = NULL, 
                        age_stride = 1, time_stride = 1, eq_surface = TRUE, alpha = 1, eq_alpha = 0.3, 
                        set_aspect = FALSE, save_html = TRUE){
  min_age <- min(density$age)
  max_age <- max(density$age)
  
  if (save_html){
    if (age_stride < 5) age_stride <- 5
    if (time_stride < 5) time_stride <- 5
  }
  
  open3d(windowRect = c(20, 30, 800, 800))
  strided_density <- plot_age_density(density, palette = rainbow, alpha = alpha, 
                                      age_stride = age_stride, time_stride = time_stride)
  
  if (eq_surface){
    eq_density <- density
    times <- unique(eq_density$time)
    for (time in times) eq_density[eq_density$time==time, 'value'] <- eq_density[eq_density$time==times[[1]], 'value']
    
    plot_age_density(eq_density, palette = colorRampPalette(c("#000000FF", "#FFFFFFFF")), alpha = eq_alpha, 
                     age_stride = age_stride, time_stride = time_stride,
                     add = TRUE)
  }
  
  legend3d("topright", legend = title, inset=c(0.02))
  
  # mean lines
  if (!is.null(mean)){
    values <- mean$value
    values[values < min_age | values > max_age] <- NaN
    if (eq_surface){
      lines3d(values, mean$time, 0, lwd=5, col="blue")

      lines3d(mean[mean$time==mean$time[[1]], 'value'], mean$time, 0, lwd = 5, col = 'black', alpha = 0.3)
    }

    mean_dv <- sapply(1:nrow(mean), function(i){
      z <- find_z_value_for_age_and_time(strided_density, values[[i]], mean$time[[i]], age_stride, time_stride)
    })
    lines3d(values, mean$time, mean_dv, lwd = 5, col = "blue")
   }
  
  # median lines
  if (!is.null(median)){
    values <- median$value
    values[values < min_age | values > max_age] <- NaN
    if (eq_surface){
      lines3d(values, median$time, 0, lwd=5, col="red")
      
      lines3d(median[median$time==median$time[[1]], 'value'], median$time, 0, lwd = 5, col = 'black', alpha = 0.3)
    }
    
    median_dv <- sapply(1:nrow(median), function(i){
      find_z_value_for_age_and_time(strided_density, values[[i]], median$time[[i]], age_stride, time_stride)
    })
    lines3d(values, median$time, median_dv, lwd=5, col="red")
  }
  
  if (set_aspect) set_aspect_ratio(density)
  
  # save html file
  if (save_html){
    if (age_stride < 5) age_stride <- 5
    if (time_stride <5) time_stride <- 5
    widget <- rglwidget()
    widget$width <- 800
    widget$height <- 800

    htmlwidgets::saveWidget(widget, paste0(gsub("\\s", "", title) , ".html"))
  }
}

########### main code ############

##### plot age densities #####

save_html = TRUE

age_density_data <- read.csv("age_dens.csv")
age_mean_data <- read.csv("age_mean.csv")
age_median_data = read.csv("age_median.csv")

pool_names <- c('Atmosphere', 'Terrestrial biosphere', 'Ocean surface layer')

for (pool in unique(age_density_data$pool)){
  age_density <- age_density_data[age_density_data$pool==pool, c('age', 'time', 'value')]

  if (pool == -1){
    title <- 'System age'
  } 
  else{
    title <- paste(pool_names[pool+1], ' age')
  }

  age_mean <- age_mean_data[age_mean_data$pool==pool, c('time', 'value')]
  age_median <- age_median_data[age_median_data$pool==pool, c('time', 'value')]

  plot_object(title, age_density, age_mean, age_median, save_html = save_html)
  #plot_object(title, age_density, age_mean, age_median, eq_surface = T, save_html = F)
  #plot_object(title, age_density, eq_surface = F, save_html = F)
}


##### plot backward transit time density #####

btt_density <- read.csv("btt_dens.csv")
btt_mean <- read.csv("btt_mean.csv")
btt_median = read.csv("btt_median.csv")

title <- 'Backward Transit Time'
plot_object(title, btt_density, btt_mean, btt_median, save_html = save_html)

title <- 'Normalized Backward Transit Time'
output_flux_data <- read.csv('output_flux.csv')
output_flux <- output_flux_data[output_flux_data$pool==-1,]
btt_density_normalized <- btt_density
btt_density_normalized$value <- btt_density$value / output_flux$value
plot_object(title, btt_density_normalized, btt_mean, btt_median, save_html = save_html)


##### plot forward transit time density #####


ftt_density <- read.csv("ftt_dens.csv")
ftt_median = read.csv("ftt_median.csv")

title <- 'Forward Transit Time'
# at time t0 ftt_density = NaN, remove it to be able to plot eq_density with value from time t0+1
ftt_density <- ftt_density[ftt_density$time!=ftt_density$time[[1]],]
ftt_median <- ftt_median[complete.cases(ftt_median),]
plot_object(title, ftt_density, median = ftt_median, save_html = save_html)

title <- 'Normalized Forward Transit Time'
input_flux_data <- read.csv('input_flux.csv')
input_flux <- input_flux_data[input_flux_data$pool==-1,]

ftt_density <- read.csv("ftt_dens.csv")
ftt_density_normalized <- ftt_density
ftt_density_normalized$value <- ftt_density$value / input_flux$value
ftt_density_normalized <- ftt_density_normalized[ftt_density_normalized$time!=ftt_density_normalized$time[[1]],]
plot_object(title, ftt_density_normalized, median = ftt_median, save_html = save_html)
