library('ggplot2')

genre_dist <- matrix(c(10.97, 26.21, 3.37, 6.07, 25.44, 27.94, 20.26, 24.32, 5.25, 7.71, 15.26, 27.2, 5.23, 1.7, 2.15, 89.2, 0.69, 1.03), ncol=6,byrow=TRUE)

rownames(genre_dist) <- c("DRAMA","COMEDY","TRAGEDY")

colnames(genre_dist) <- c("0",'1', '2', '3', '4', '5')

barplot(genre_dist, main="Topics distribution in genre", xlab="Topics", ylab="Probability", col=c("grey","orange", "black"), beside=TRUE)

legend("topright", legend=rownames(genre_dist), cex=0.5, fill=c("grey","orange", "black"))