# replace with a directory on your machine
path <- "/Users/Jake/Desktop/"


# define helper functions ----------------------------------------


# from the FIAR package
# https://github.com/cran/FIAR/blob/master/R/hrfConvolve.R
hrfConvolve <-
  function (x=NULL, scans = NA, onsets = c(), durations = c(), rt = NA, 
            SNR=0, mean = FALSE, a1 = 6, a2 = 12, b1 = 0.9, b2 = 0.9, 
            cc = 0.2) 
  {
    hrf <- function(x, a1, a2, b1, b2, c) {
      d1 <- a1 * b1
      d2 <- a2 * b2
      c1 <- (x/d1)^a1
      c2 <- c * (x/d2)^a2
      res <- c1 * exp(-(x - d1)/b1) - c2 * exp(-(x - d2)/b2)
      res
    }
    if (is.null(x)){ 
      numberofonsets <- length(onsets)
      if (length(durations) == 1) {
        durations <- rep(durations, numberofonsets)
      }
      
      stimulus <- rep(0, scans)
      for (i in 1:numberofonsets) {
        for (j in onsets[i]:(onsets[i] + durations[i] - 1)) {
          stimulus[j] <- 1
        }
      }
      hrfnn <- convolve(stimulus, hrf(scans:1, a1, a2, b1/rt,b2/rt, cc))
      
      if(SNR>0){
        sdS <- sd(hrfnn)	
        noise=rnorm(scans,sd=sdS/SNR)
        #Zx <- x/sd(x)
        #sdN <- sdS/SNR
        hrfnn <- hrfnn + noise
      }
      else{hrfnn <- hrfnn}
      
      if (mean) {
        hrfnn - mean(hrfnn)
      }
      else {
        hrfnn
      }
    }
    else{ hrfnn <- convolve(x, hrf(length(x):1, a1, a2, b1,b2, cc))
    
    if(SNR>0){
      sdS <- sd(hrfnn)	
      noise <- rnorm(length(x),sd=sdS/SNR)
      
      hrfnn <- hrfnn + noise
    }
    else{hrfnn <- hrfnn}
    
    if (mean) {
      hrfnn - mean(hrfnn)
    }
    else {
      hrfnn
    }
    }
    
  }

# function to build activation sequence
act <- function(num_trials, pad=25, ISI=1:3, stim_sd=0){
  if(length(ISI)==1) ISI <- rep(ISI,2)
  ans <- c(unlist(lapply(seq(num_trials), function(x){
    c(1, rep(0, sample(ISI, 1)))
  })), numeric(pad))
  ans[ans>0] <- 1 + stim_sd*scale(rnorm(sum(ans>0)))
  return(ans)
  # return(ans * (rnorm(length(ans), mean=1, sd=stim_sd)))
}

# function to draw a single panel w/ specified parameters
hrf_plot <- function(means=c(1,1), stim_sd=0, main=""){
  act1 <- means[1]*head(c(act(5, ISI=3, stim_sd=stim_sd), numeric(15), act(5, ISI=3, stim_sd=stim_sd), numeric(30)), -15)
  act2 <- means[2]*head(c(numeric(30), act(5, ISI=3, stim_sd=stim_sd), numeric(15), act(5, ISI=3, stim_sd=stim_sd)), -15)
  plot(y=c(-.25,1.9), x=c(0,length(act1)), cex=0, xlab="", xaxt="n",
       yaxt="n", ylab="Neural activation\n(arbitrary scale)",
       mgp=c(1,1,0), main=main)
  axis(side=1, tick=FALSE, line=-1)
  lines(y=rep(act1, each=10), x=seq(length(act1)*10)/10, col="red")
  lines(y=rep(act2, each=10), x=seq(length(act2)*10)/10, col="blue")
  lines(hrfConvolve(x=act1, rt=1), type="l", col="red")
  conv2 <- hrfConvolve(x=act2, rt=1)
  conv2[1:20] <- 0
  lines(conv2, type="l", col="blue")
  abline(h=means, lty=2:3)
}


# build the plot! ---------------------------------------------------------


png(paste0(path,"time_series.png"), height=10, width=8,
    units="in", res=200, pointsize=19)
layout(cbind(c(rep(1:3, each=4),7,7,rep(4:6, each=4))))
par(mar=c(.5,3,1,1)+.1)
par(oma=c(2,0,2,0))
x <- c(1,5,9,13,17,31,35,39,43,47,61,65,69,73,77,91,95,99,103,107)
labs <- c("chair","house","tree","desk","spoon",
          "run","pay","speak,","climb","read",
          "road","paper","bread","mug","sign",
          "eat","take","see","find","ask")

hrf_plot(means=c(1.25,.75), stim_sd=0, main="Participant 1")
text(x=c(40), y=1.7, cex=.75,
     labels="Stimulus presentations")
arrows(x0=c(37,52), x1=c(31,59), length=.05,
       y0=1.58, y1=c(.8, 1.27), lwd=.8)
hrf_plot(means=c(1.1,.9), stim_sd=0, main="Participant 2")
text(x=x, y=c(1.8,-.15), labels=labs, cex=.75,
     col=rep(c("red","blue"),each=5))
hrf_plot(means=c(1.4,.6), stim_sd=0, main="Participant 3")

set.seed(45634)
hrf_plot(means=c(1.25,.75), stim_sd=.25, main="Participant 1")
text(x=43, y=1.78, cex=.75,
     labels="Variable activations estimated from the data")
arrows(x0=c(38,55), x1=c(35,60), length=.05,
       y0=1.65, y1=c(.95, 1.35), lwd=.8)
set.seed(45634)
hrf_plot(means=c(1.1,.9), stim_sd=.25, main="Participant 2")
text(x=x, y=c(1.8,-.15), labels=labs, cex=.75,
     col=rep(c("red","blue"),each=5))
set.seed(45634)
hrf_plot(means=c(1.4,.6), stim_sd=.25, main="Participant 3")

mtext("Time (in seconds)", side=1, line=.5, outer=TRUE, cex=.9)
mtext("A: Standard model (stimulus effects ignored)", side=3, line=.25, outer=TRUE)
mtext("Time (in seconds)", side=3, line=16.5, cex=.9)
mtext("B: Random stimulus model", side=3, line=14.5)
dev.off()


