

import matplotlib.pyplot as plt

# # Data
# # stop_thresholds = [2.5, 3, 3.5]
# # precision = [0.13697846578486128, 0.15647378513618243, 0.09780676641394509]
# # recall = [0.07722385496572037, 0.08653615769152401, 0.07297620473793758]
# # f_score = [0.0987664852381698, 0.11144103807089585, 0.0835863969626343]






# # # Data
# # stop_thresholds = [2.5, 3, 3.5]
# # precision = [0.08747763303626985, 0.13290925238783324, 0.08611048060749145]
# # recall = [0.0765298958863195, 0.08348411919950907, 0.07902868899614829]
# # f_score = [0.08163837590416029, 0.10255223427289621, 0.0824177378095319]

# # N = [10, 30, 50]
# # precision = [0.08578520704695922, 0.07745555491438975, 0.09768532124415308]
# # recall= [0.0492830390901482, 0.051975539233374765, 0.062406832165513765]
# # f_score = [0.06260177107741763,0.062207528411984916,0.07615902863544152]



# locality = [0,2,10]
# precision =[ 0.09169925732609387, 0.11078658863123363,0.11161050855885528]
# recall = [0.09104348194593109, 0.06958823862772362,0.07130301857872041]
# f_score = [0.09137019300554569,0.08548247761150822, 0.08701561103643116]



# # Plot
# plt.figure(figsize=(10, 6))

# plt.plot(locality, precision, marker='o', label='Precision')
# plt.plot(locality, recall, marker='o', label='Recall')
# plt.plot(locality, f_score, marker='o', label='F-score')

# plt.title('Performance Metrics vs locality')
# plt.xlabel('locality')
# plt.ylabel('Score')
# plt.xticks(locality)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# plt.show()

# x = [326, 261, 225]
# plt.plot([10,30,50],x,marker='o')
# plt.xlabel("N")
# plt.ylabel("number of fails")
# plt.ylim(0)
# plt.savefig("nfails.png")
# plt.show()



N = [10,10,30,30,50,50]
maks = [2,10,1,1,1,1]
# sumi = [31,408,10,16,17,10]
plt.scatter(N,maks,label="max_centraliziran")




maks = [3,2,1,1,1,1]
plt.scatter(N,maks,label="max_enakomerno_naključen",marker="x")
plt.legend()
plt.ylabel("Maksimalno št. zaporednih neuspehov")
plt.xlabel("Število kandidatov")
plt.xticks(N)
plt.ylim(0)

plt.savefig("Strike.png")
plt.show()




# sumi = [12,32,7,18,15,18]