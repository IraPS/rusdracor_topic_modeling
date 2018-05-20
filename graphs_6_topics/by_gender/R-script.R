gender_dist <- matrix(c(19.89, 25.64, 7.03, 16.36, 13.55, 17.54, 12.32, 15.9, 10.25, 23.56, 15.7, 22.27, 15.82, 14.52, 12.84, 30.21, 14.59, 12.02),ncol=6,byrow=TRUE)

rownames(gender_dist) <- c("FEMALE","MALE","UNKNOWN")

colnames(gender_dist) <- c("0",'1', '2', '3', '4', '5')

barplot(gender_dist, main="Topics distribution in gender", xlab="Topics", ylab="Probability", col=c("pink","blue", "grey"), beside=TRUE)

legend("top", legend=rownames(gender_dist), cex=0.5, fill=c("pink","blue", "grey"))