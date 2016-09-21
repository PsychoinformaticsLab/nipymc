
# get simulation data -----------------------------------------------------


test_stats <- read.table("test_stats.txt")
test_stats <- test_stats[,-1]
names(test_stats) <- c("model","sim_SD","sim_mean","type")
test_stats <- data.frame(test_stats,
                         expand.grid(dummy=1:7,
                                     s=0:2, q=c(16,32,64),
                                     p=c(16,32,64))[,-1])


# s=0 ---------------------------------------------------------------------


png("z_s0.png", units="in", height=6, width=6, res=200, pointsize=15)
layout(matrix(1:9, ncol=3))
par(mar=c(2.5, 3, .5, .5)+.1)
par(oma=c(.5,0,3,2))
condition <- with(test_stats, s==0 & substr(model,1,6) != "nipymc")
by(test_stats[condition,], list(test_stats[condition,]$p, test_stats[condition,]$q), function(panel){
  ylim <- with(test_stats[condition,], range(c(sim_mean - sim_SD, sim_mean + sim_SD)))
  plot(y=panel$sim_mean, x=1:4, type="o", ylim=ylim, mgp=2:0,
       ylab="Test statistic (z)", xlab="", xaxt="n")
  axis(side=1, at=1:4, labels=c("SPM","FSM","NSM","RSM"), las=3)
  segments(x0=1:4, x1=1:4, y0=panel$sim_mean - panel$sim_SD, y1=panel$sim_mean + panel$sim_SD)
})
mtext(side=3, at=c(.2,.53,.87), outer=TRUE, line=.1, cex=1,
      text=paste("m =",c(16,32,64)))
mtext(side=4, at=c(.2,.53,.87), outer=TRUE, line=.25, cex=1,
      text=paste("n =",c(64,32,16)))
mtext(side=3, at=.53, outer=TRUE, line=1.5, cex=1,
      text=expression(paste("Test statistics: ",sigma[Stim]," = 0")))
dev.off()

# s=1 ---------------------------------------------------------------------


png("z_s1.png", units="in", height=6, width=6, res=200, pointsize=15)
layout(matrix(1:9, ncol=3))
par(mar=c(2.5, 3, .5, .5)+.1)
par(oma=c(.5,0,3,2))
condition <- with(test_stats, s==1 & substr(model,1,6) != "nipymc")
by(test_stats[condition,], list(test_stats[condition,]$p, test_stats[condition,]$q), function(panel){
  ylim <- with(test_stats[condition,], range(c(sim_mean - sim_SD, sim_mean + sim_SD)))
  plot(y=panel$sim_mean, x=1:4, type="o", ylim=ylim, mgp=2:0,
       ylab="Test statistic (z)", xlab="", xaxt="n")
  axis(side=1, at=1:4, labels=c("SPM","FSM","NSM","RSM"), las=3)
  segments(x0=1:4, x1=1:4, y0=panel$sim_mean - panel$sim_SD, y1=panel$sim_mean + panel$sim_SD)
})
mtext(side=3, at=c(.2,.53,.87), outer=TRUE, line=.1, cex=1,
      text=paste("m =",c(16,32,64)))
mtext(side=4, at=c(.2,.53,.87), outer=TRUE, line=.25, cex=1,
      text=paste("n =",c(64,32,16)))
mtext(side=3, at=.53, outer=TRUE, line=1.5, cex=1,
      text=expression(paste("Test statistics: ",sigma[Stim]," = 1")))
dev.off()

# s=2 ---------------------------------------------------------------------


png("z_s2.png", units="in", height=6, width=6, res=200, pointsize=15)
layout(matrix(1:9, ncol=3))
par(mar=c(2.5, 3, .5, .5)+.1)
par(oma=c(.5,0,3,2))
condition <- with(test_stats, s==2 & substr(model,1,6) != "nipymc")
by(test_stats[condition,], list(test_stats[condition,]$p, test_stats[condition,]$q), function(panel){
  ylim <- with(test_stats[condition,], range(c(sim_mean - sim_SD, sim_mean + sim_SD)))
  plot(y=panel$sim_mean, x=1:4, type="o", ylim=ylim, mgp=2:0,
       ylab="Test statistic (z)", xlab="", xaxt="n")
  axis(side=1, at=1:4, labels=c("SPM","FSM","NSM","RSM"), las=3)
  segments(x0=1:4, x1=1:4, y0=panel$sim_mean - panel$sim_SD, y1=panel$sim_mean + panel$sim_SD)
})
mtext(side=3, at=c(.2,.53,.87), outer=TRUE, line=.1, cex=1,
      text=paste("m =",c(16,32,64)))
mtext(side=4, at=c(.2,.53,.87), outer=TRUE, line=.25, cex=1,
      text=paste("n =",c(64,32,16)))
mtext(side=3, at=.53, outer=TRUE, line=1.5, cex=1,
      text=expression(paste("Test statistics: ",sigma[Stim]," = 2")))
dev.off()
