gender_dist <- matrix(c(25.92, 27.12, 10.61, 20.81, 15.55, 27.24, 15.13, 13.85, 25.82, 17.96, 16.99, 16.78, 17.3, 30.58, 18.35),ncol=7,byrow=TRUE)

rownames(gender_dist) <- c("FEMALE","MALE","UNKNOWN")

colnames(gender_dist) <- c("0",'1', '2', '3', '4', '5', '6')

barplot(gender_dist, main="Topics distribution in gender", xlab="Topics", ylab="Probability", col=c("pink","blue", "grey"), beside=TRUE)

legend("top", legend=rownames(gender_dist), cex=0.5, fill=c("pink","blue", "grey"))